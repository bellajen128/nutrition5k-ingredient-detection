#!/usr/bin/env python3
"""
Week 7: EfficientDet-Based Multi-Label Ingredient Classification
Dataset: Nutrition5K
Environment: Discovery HPC with V100 GPU
Time limit: 1 hour (55 min training + 5 min buffer)

Author: Chen Jen 
Date: 2025-11-06
"""

import os
import sys
import time
import json
import signal
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Step 0: Environment Configuration
# =============================================================================

print("="*80)
print("WEEK 7: EfficientDet Training for Nutrition5K Ingredient Detection")
print("="*80)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Virtual environment setup
VENV_DIR = "/home/jen.che/nutrition5k_env"
SITE_PACKAGES = f"{VENV_DIR}/lib/python3.12/site-packages"
if SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SITE_PACKAGES)

# Time management for 1-hour GPU session
START_TIME = time.time()
TIME_LIMIT_SECONDS = 55 * 60  # 55 minutes with 5-min buffer

def check_time_remaining():
    """Returns (has_time: bool, remaining_seconds: float)"""
    elapsed = time.time() - START_TIME
    remaining = TIME_LIMIT_SECONDS - elapsed
    return remaining > 0, remaining

def save_emergency_checkpoint(model, optimizer, scheduler, history, epoch):
    """Save checkpoint when time runs out"""
    emergency_path = MODEL_DIR / f'emergency_epoch{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }, emergency_path)
    print(f"\nEmergency checkpoint saved: {emergency_path}")

# =============================================================================
# Step 1: Import Required Libraries
# =============================================================================

print("="*80)
print("Importing Libraries")
print("="*80)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ All libraries imported successfully")

# =============================================================================
# Step 2: Hardware Configuration
# =============================================================================

print("\n" + "="*80)
print("Hardware Configuration")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
    
    # Enable optimizations for V100
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print("✓ cuDNN optimizations enabled")
else:
    print("⚠️  WARNING: CUDA not available!")
    print("   This script should be run on GPU node via SLURM")
    print("   Continuing anyway for testing purposes...")

# =============================================================================
# Step 3: Directory Structure
# =============================================================================

print("\n" + "="*80)
print("Directory Setup")
print("="*80)

BASE_DIR = Path("/scratch/jen.che/nutrition5k_prepared")
IMG_DIR = BASE_DIR / "images"

# Output directories
OUTPUT_DIR = BASE_DIR / "efficientdet_outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
VIS_DIR = OUTPUT_DIR / "visualizations"

# Create all directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, VIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"  ├─ models/")
print(f"  ├─ logs/")
print(f"  └─ visualizations/")

# =============================================================================
# Step 4: Load Dataset and Vocabulary
# =============================================================================

print("\n" + "="*80)
print("Loading Dataset")
print("="*80)

# Load ingredient vocabulary
vocab_path = BASE_DIR / "ingredient_vocab.json"
if not vocab_path.exists():
    raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

with open(vocab_path) as f:
    vocab_data = json.load(f)
    vocab = vocab_data['vocab']

num_classes = len(vocab)
print(f"✓ Ingredient vocabulary loaded: {num_classes} classes")
print(f"  Sample ingredients: {vocab[:5]}")

# Load CSV files
train_csv = BASE_DIR / "train.csv"
val_csv = BASE_DIR / "val.csv"
test_csv = BASE_DIR / "test.csv"

for csv_path in [train_csv, val_csv, test_csv]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

print(f"\n✓ Dataset splits loaded:")
print(f"    Train: {len(train_df):,} samples")
print(f"    Val:   {len(val_df):,} samples")
print(f"    Test:  {len(test_df):,} samples")
print(f"    Total: {len(train_df) + len(val_df) + len(test_df):,} samples")

# Analyze label distribution
print("\n✓ Dataset statistics:")
sample_labels = json.loads(train_df.iloc[0]['label_json'])
avg_ingredients = train_df['label_json'].apply(
    lambda x: sum(json.loads(x))
).mean()
print(f"    Average ingredients per dish: {avg_ingredients:.2f}")

# =============================================================================
# Step 5: Dataset Class Definition
# =============================================================================

class Nutrition5KDataset(Dataset):
    """
    Multi-label classification dataset for Nutrition5K
    Returns: (image_tensor, label_vector, dish_id)
    """
    
    def __init__(self, df, transform=None, img_size=512):
        """
        Args:
            df: DataFrame with columns [dish_id, image_path, label_json]
            transform: torchvision transforms
            img_size: target image size (default 512)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        
        # Pre-parse all labels for faster access
        self.labels_cache = []
        for idx in range(len(self.df)):
            label_json = self.df.iloc[idx]['label_json']
            labels = np.array(json.loads(label_json), dtype=np.float32)
            self.labels_cache.append(labels)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback: create blank image
            print(f"Warning: Failed to load {img_path}, using blank image")
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')
        
        # Get pre-parsed labels
        labels = self.labels_cache[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(labels, dtype=torch.float32)

# =============================================================================
# Step 6: Data Augmentation & Transforms
# =============================================================================

print("\n" + "="*80)
print("Data Augmentation Setup")
print("="*80)

IMG_SIZE = 512  # Optimal for V100 GPU

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/Test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print(f"✓ Image size: {IMG_SIZE}×{IMG_SIZE}")
print(f"✓ Train augmentation:")
print(f"    - Horizontal flip (p=0.5)")
print(f"    - Random rotation (±10°)")
print(f"    - Color jitter (brightness/contrast/saturation)")

# Create datasets
print("\nCreating datasets...")
train_dataset = Nutrition5KDataset(train_df, train_transform, IMG_SIZE)
val_dataset = Nutrition5KDataset(val_df, val_transform, IMG_SIZE)
test_dataset = Nutrition5KDataset(test_df, val_transform, IMG_SIZE)
print("✓ Datasets created")

# Test dataset
print("\nTesting dataset...")
sample_img, sample_labels = train_dataset[0]
print(f"  Sample image shape: {sample_img.shape}")
print(f"  Sample labels shape: {sample_labels.shape}")
print(f"  Num ingredients in sample: {int(sample_labels.sum())}")

# =============================================================================
# Step 7: Model Architecture
# =============================================================================

print("\n" + "="*80)
print("Model Architecture")
print("="*80)

class EfficientNetMultiLabel(nn.Module):
    """
    EfficientNet-B0 backbone + Multi-label classification head
    
    Architecture:
    - EfficientNet-B0 backbone (pretrained on ImageNet)
    - Global Average Pooling
    - Fully connected layers with dropout
    - Final sigmoid activation for multi-label output
    """
    
    def __init__(self, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        
        # Load EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove original classification head
            global_pool=''  # Remove global pooling (we'll add custom)
        )
        
        # Get backbone output dimension (EfficientNet-B0: 1280)
        self.feature_dim = 1280
        
        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # Global average pooling
            nn.Flatten(),                      # Flatten to vector
            nn.Dropout(dropout),               # Regularization
            nn.Linear(self.feature_dim, 512), # First FC layer
            nn.ReLU(inplace=True),            # Activation
            nn.Dropout(dropout / 2),          # Less dropout in second layer
            nn.Linear(512, num_classes)       # Output layer
        )
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input images [batch_size, 3, H, W]
        Returns:
            logits: [batch_size, num_classes]
        """
        features = self.backbone(x)  # [B, 1280, H', W']
        logits = self.classifier(features)  # [B, num_classes]
        return logits

# Initialize model
print("Initializing EfficientNet-B0...")
model = EfficientNetMultiLabel(
    num_classes=num_classes,
    pretrained=True,
    dropout=0.3
)
model = model.to(device)

# Calculate model size
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✓ Model: EfficientNet-B0 Multi-Label Classifier")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1e6:.1f} MB (float32)")

# =============================================================================
# Step 8: Training Configuration
# =============================================================================

print("\n" + "="*80)
print("Training Configuration")
print("="*80)

# Hyperparameters (optimized for V100 + 1 hour session)
BATCH_SIZE = 32           # V100 can handle this
NUM_EPOCHS = 20          # Should complete in ~40-45 minutes
LEARNING_RATE = 1e-3     # Higher LR for faster convergence
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 8          # HPC has many cores
LABEL_THRESHOLD = 0.5    # For binary prediction

config = {
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'num_workers': NUM_WORKERS,
    'img_size': IMG_SIZE,
    'num_classes': num_classes,
    'label_threshold': LABEL_THRESHOLD,
    'device': str(device)
}

print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Workers: {NUM_WORKERS}")
print(f"Label threshold: {LABEL_THRESHOLD}")

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True,
    drop_last=True  # Drop incomplete batches
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"\n✓ DataLoaders created:")
print(f"    Train batches: {len(train_loader)}")
print(f"    Val batches: {len(val_loader)}")
print(f"    Test batches: {len(test_loader)}")
print(f"    Estimated time per epoch: ~{len(train_loader) * BATCH_SIZE / 1000:.1f} minutes")

# =============================================================================
# Step 9: Loss Function, Optimizer, Scheduler
# =============================================================================

print("\n" + "="*80)
print("Training Components")
print("="*80)

# Loss function for multi-label classification
criterion = nn.BCEWithLogitsLoss()
print("✓ Loss function: BCEWithLogitsLoss")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
print(f"✓ Optimizer: AdamW (lr={LEARNING_RATE})")

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-6
)
print(f"✓ Scheduler: CosineAnnealingLR")

# =============================================================================
# Step 10: Training & Validation Functions
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Train model for one epoch
    Returns: (avg_loss, early_stop_flag)
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(
        loader,
        desc=f'Epoch {epoch+1:02d}/{NUM_EPOCHS} [TRAIN]',
        ncols=100
    )
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Check time remaining
        has_time, remaining = check_time_remaining()
        if not has_time or remaining < 300:  # Need 5 min buffer
            print(f"\nTime limit approaching! ({remaining/60:.1f} min left)")
            return running_loss / (batch_idx + 1), True
        
        # Move to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg': f'{avg_loss:.4f}'
        })
    
    epoch_loss = running_loss / len(loader)
    return epoch_loss, False

def validate_one_epoch(model, loader, criterion, device, epoch, phase='VAL'):
    """
    Validate model for one epoch
    Returns: (avg_loss, predictions, ground_truth_labels)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(
            loader,
            desc=f'Epoch {epoch+1:02d}/{NUM_EPOCHS} [{phase}]  ',
            ncols=100
        )
        
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get probabilities
            probs = torch.sigmoid(outputs)
            
            # Collect for metrics
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return epoch_loss, all_preds, all_labels

def compute_metrics(preds, labels, threshold=0.5):
    """
    Compute multi-label classification metrics
    
    Metrics:
    - Sample-wise precision/recall/f1 (average over samples)
    - Exact match ratio (all labels correct)
    - Hamming accuracy (per-label accuracy)
    """
    preds_binary = (preds >= threshold).astype(int)
    
    # Sample-wise metrics
    precision = precision_score(labels, preds_binary, average='samples', zero_division=0)
    recall = recall_score(labels, preds_binary, average='samples', zero_division=0)
    f1 = f1_score(labels, preds_binary, average='samples', zero_division=0)
    
    # Exact match ratio
    exact_match = accuracy_score(labels, preds_binary)
    
    # Hamming accuracy (per-label)
    hamming_acc = (labels == preds_binary).mean()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': exact_match,
        'hamming_accuracy': hamming_acc
    }

# =============================================================================
# Step 11: Training Loop
# =============================================================================

print("\n" + "="*80)
print("TRAINING LOOP")
print("="*80)
print(f"Time budget: {TIME_LIMIT_SECONDS/60:.0f} minutes")
print(f"Target epochs: {NUM_EPOCHS}\n")

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': [],
    'val_exact_match': [],
    'val_hamming_acc': [],
    'learning_rate': []
}

best_val_loss = float('inf')
best_val_f1 = 0.0
best_model_path = MODEL_DIR / 'efficientnet_best.pth'

# Training loop with error handling
try:
    for epoch in range(NUM_EPOCHS):
        # Check time
        has_time, remaining = check_time_remaining()
        if not has_time or remaining < 600:  # Need 10 min for epoch
            print(f"\nInsufficient time for next epoch ({remaining/60:.1f} min left)")
            print("Saving final checkpoint and exiting...")
            save_emergency_checkpoint(model, optimizer, scheduler, history, epoch)
            break
        
        # Epoch header
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
        print(f"Time remaining: {remaining/60:.1f} minutes")
        print('='*80)
        
        # Training phase
        train_loss, early_stop = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        history['train_loss'].append(train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        if early_stop:
            print("Early stop triggered - saving checkpoint...")
            save_emergency_checkpoint(model, optimizer, scheduler, history, epoch)
            break
        
        # Validation phase
        val_loss, val_preds, val_labels = validate_one_epoch(
            model, val_loader, criterion, device, epoch, phase='VAL'
        )
        history['val_loss'].append(val_loss)
        
        # Compute metrics
        metrics = compute_metrics(val_preds, val_labels, threshold=LABEL_THRESHOLD)
        history['val_precision'].append(metrics['precision'])
        history['val_recall'].append(metrics['recall'])
        history['val_f1'].append(metrics['f1'])
        history['val_exact_match'].append(metrics['exact_match'])
        history['val_hamming_acc'].append(metrics['hamming_accuracy'])
        
        # Learning rate step
        scheduler.step()
        
        # Print epoch summary
        print(f"\n{'─'*80}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print('─'*80)
        print(f"  Train Loss:      {train_loss:.4f}")
        print(f"  Val Loss:        {val_loss:.4f}")
        print(f"  Precision:       {metrics['precision']:.4f}")
        print(f"  Recall:          {metrics['recall']:.4f}")
        print(f"  F1-Score:        {metrics['f1']:.4f}")
        print(f"  Exact Match:     {metrics['exact_match']:.4f}")
        print(f"  Hamming Acc:     {metrics['hamming_accuracy']:.4f}")
        print(f"  Learning Rate:   {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model (based on F1 score)
        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics,
                'history': history,
                'config': config
            }
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ Best model saved! (F1: {metrics['f1']:.4f})")
        
        # Also track best loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            checkpoint_path = MODEL_DIR / f'checkpoint_epoch{epoch+1:02d}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint saved: epoch {epoch+1}")

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user (Ctrl+C)")
    save_emergency_checkpoint(model, optimizer, scheduler, history, epoch)
except Exception as e:
    print(f"\n\nError during training: {e}")
    import traceback
    traceback.print_exc()
    save_emergency_checkpoint(model, optimizer, scheduler, history, epoch)

# =============================================================================
# Step 12: Final Model Evaluation on Test Set
# =============================================================================

print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80)

if best_model_path.exists():
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    print(f"  Best val F1: {checkpoint['metrics']['f1']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_preds, test_labels = validate_one_epoch(
        model, test_loader, criterion, device, epoch=0, phase='TEST'
    )
    
    test_metrics = compute_metrics(test_preds, test_labels, threshold=LABEL_THRESHOLD)
    
    print(f"\n{'─'*80}")
    print("TEST SET RESULTS")
    print('─'*80)
    print(f"  Test Loss:       {test_loss:.4f}")
    print(f"  Precision:       {test_metrics['precision']:.4f}")
    print(f"  Recall:          {test_metrics['recall']:.4f}")
    print(f"  F1-Score:        {test_metrics['f1']:.4f}")
    print(f"  Exact Match:     {test_metrics['exact_match']:.4f}")
    print(f"  Hamming Acc:     {test_metrics['hamming_accuracy']:.4f}")
else:
    print("⚠️  No best model found - skipping test evaluation")
    test_metrics = None

# =============================================================================
# Step 13: Save All Results
# =============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# 1. Save final model
final_model_path = MODEL_DIR / 'efficientnet_final.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'history': history
}, final_model_path)
print(f"✓ Final model: {final_model_path}")

# 2. Save training history
history_df = pd.DataFrame(history)
history_df.to_csv(LOG_DIR / 'training_history.csv', index=False)
print(f"✓ Training history: {LOG_DIR / 'training_history.csv'}")

# 3. Save comprehensive summary
total_time_minutes = (time.time() - START_TIME) / 60
summary = {
    'metadata': {
        'training_date': time.strftime('%Y-%m-%d'),
        'training_time_minutes': round(total_time_minutes, 2),
        'epochs_completed': len(history['train_loss']),
        'target_epochs': NUM_EPOCHS,
        'early_stopped': len(history['train_loss']) < NUM_EPOCHS
    },
    'config': config,
    'best_validation': {
        'epoch': checkpoint['epoch'] + 1 if best_model_path.exists() else None,
        'val_loss': float(best_val_loss),
        'f1_score': float(best_val_f1)
    },
    'final_validation': {
        'loss': float(history['val_loss'][-1]) if history['val_loss'] else None,
        'precision': float(history['val_precision'][-1]) if history['val_precision'] else None,
        'recall': float(history['val_recall'][-1]) if history['val_recall'] else None,
        'f1': float(history['val_f1'][-1]) if history['val_f1'] else None
    },
    'test_set': test_metrics if test_metrics else None
}

with open(OUTPUT_DIR / 'training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Training summary: {OUTPUT_DIR / 'training_summary.json'}")

# 4. Generate training curves
if len(history['train_loss']) > 1:
    print("\nGenerating training curves...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create 2x3 subplot grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2, markersize=5)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation', linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Curves', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Precision
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['val_precision'], 'g-o', linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Validation Precision', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Recall
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['val_recall'], 'orange', marker='o', linewidth=2, markersize=5)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Recall', fontsize=11)
    ax3.set_title('Validation Recall', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # F1 Score
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history['val_f1'], 'purple', marker='o', linewidth=2, markersize=5)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('F1-Score', fontsize=11)
    ax4.set_title('Validation F1-Score', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # Exact Match
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, history['val_exact_match'], 'brown', marker='o', linewidth=2, markersize=5)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Exact Match Ratio', fontsize=11)
    ax5.set_title('Exact Match Accuracy', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # Learning Rate
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, history['learning_rate'], 'teal', marker='o', linewidth=2, markersize=5)
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Learning Rate', fontsize=11)
    ax6.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    plt.suptitle('EfficientNet-B0 Training Progress - Nutrition5K Ingredient Detection',
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(VIS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(VIS_DIR / 'training_curves.pdf', bbox_inches='tight')
    print(f"✓ Training curves: {VIS_DIR / 'training_curves.png'}")

# =============================================================================
# Step 14: Final Summary Report
# =============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nTraining completed in {total_time_minutes:.1f} minutes")
print(f"Epochs: {len(history['train_loss'])}/{NUM_EPOCHS}")

if history['train_loss']:
    print(f"\nBest Results:")
    print(f"    Best Val Loss: {best_val_loss:.4f}")
    print(f"    Best Val F1:   {best_val_f1:.4f}")
    
    print(f"\nFinal Metrics:")
    print(f"    Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"    Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"    Precision:  {history['val_precision'][-1]:.4f}")
    print(f"    Recall:     {history['val_recall'][-1]:.4f}")
    print(f"    F1-Score:   {history['val_f1'][-1]:.4f}")

if test_metrics:
    print(f"\nTest Set Performance:")
    print(f"    Precision:  {test_metrics['precision']:.4f}")
    print(f"    Recall:     {test_metrics['recall']:.4f}")
    print(f"    F1-Score:   {test_metrics['f1']:.4f}")

print(f"\nOutput Files:")
print(f"    {OUTPUT_DIR}/")
print(f"    ├─ models/efficientnet_best.pth")
print(f"    ├─ models/efficientnet_final.pth")
print(f"    ├─ logs/training_history.csv")
print(f"    ├─ visualizations/training_curves.png")
print(f"    └─ training_summary.json")

print("\n" + "="*80)
print("WEEK 7 TRAINING COMPLETE! 🎉")
print("="*80)
print("\nNext steps:")
print("  1. Review training_curves.png for convergence")
print("  2. Check training_summary.json for metrics")
print("  3. Proceed to Week 8: Model evaluation and visualization")