#!/usr/bin/env python3
"""
Week 8: Detailed Model Evaluation
Comprehensive analysis of EfficientNet performance on Nutrition5K
"""

import os
import sys
from pathlib import Path

# Virtual environment setup
VENV_DIR = "/home/jen.che/nutrition5k_env"
SITE_PACKAGES = f"{VENV_DIR}/lib/python3.12/site-packages"
if SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SITE_PACKAGES)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
import timm
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss

print("="*80)
print("Week 8: Detailed Model Evaluation")
print("="*80)

# Check environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
print(f"PyTorch: {torch.__version__}")

# Paths
BASE_DIR = Path("/scratch/jen.che/nutrition5k_prepared")
OUTPUT_DIR = BASE_DIR / "efficientdet_outputs"
EVAL_DIR = OUTPUT_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

print(f"Output directory: {EVAL_DIR}")

# ============================================================================
# Load Data
# ============================================================================

print("\n" + "="*80)
print("Loading Data")
print("="*80)

with open(BASE_DIR / "ingredient_vocab.json") as f:
    vocab = json.load(f)['vocab']
num_classes = len(vocab)

test_df = pd.read_csv(BASE_DIR / "test.csv")

print(f"✓ Vocabulary: {num_classes} classes")
print(f"✓ Test set: {len(test_df)} samples")

# ============================================================================
# Dataset
# ============================================================================

class Nutrition5KDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        print("Pre-parsing labels...")
        self.labels_cache = [
            np.array(json.loads(row['label_json']), dtype=np.float32)
            for _, row in tqdm(df.iterrows(), total=len(df), desc='Loading')
        ]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except:
            img = Image.new('RGB', (512, 512), 'black')
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(self.labels_cache[idx]), row['dish_id']

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = Nutrition5KDataset(test_df, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ============================================================================
# Model
# ============================================================================

print("\n" + "="*80)
print("Loading Model")
print("="*80)

class EfficientNetMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='')
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

model = EfficientNetMultiLabel(num_classes).to(device)

checkpoint_path = OUTPUT_DIR / "models" / "efficientnet_best.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")
print(f"  Val F1: {checkpoint['metrics']['f1']:.4f}")

# ============================================================================
# Generate Predictions
# ============================================================================

print("\n" + "="*80)
print("Generating Predictions")
print("="*80)

all_preds = []
all_labels = []
all_ids = []

with torch.no_grad():
    for imgs, labels, ids in tqdm(test_loader, desc='Predicting'):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        
        all_preds.append(probs)
        all_labels.append(labels.numpy())
        all_ids.extend(ids)

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
preds_binary = (all_preds >= 0.5).astype(int)

print(f"✓ Predictions: {all_preds.shape}")

# ============================================================================
# Overall Metrics
# ============================================================================

print("\n" + "="*80)
print("Computing Metrics")
print("="*80)

overall_metrics = {
    'sample_f1': float(f1_score(all_labels, preds_binary, average='samples', zero_division=0)),
    'sample_precision': float(precision_score(all_labels, preds_binary, average='samples', zero_division=0)),
    'sample_recall': float(recall_score(all_labels, preds_binary, average='samples', zero_division=0)),
    'exact_match': float(accuracy_score(all_labels, preds_binary)),
    'hamming_accuracy': float(1 - hamming_loss(all_labels, preds_binary))
}

print("\nTest Set Performance:")
for metric, value in overall_metrics.items():
    print(f"  {metric:20s}: {value:.4f}")

with open(EVAL_DIR / 'overall_metrics.json', 'w') as f:
    json.dump(overall_metrics, f, indent=2)

# ============================================================================
# Per-Class Analysis
# ============================================================================

p, r, f1, sup = precision_recall_fscore_support(all_labels, preds_binary, average=None, zero_division=0)

per_class_df = pd.DataFrame({
    'ingredient': vocab,
    'precision': p,
    'recall': r,
    'f1_score': f1,
    'support': sup.astype(int)
}).sort_values('support', ascending=False)

print("\n" + "="*60)
print("Top 20 Ingredients by Support")
print("="*60)
print(per_class_df.head(20).to_string(index=False))

per_class_df.to_csv(EVAL_DIR / 'per_class_metrics.csv', index=False)

# Best and worst
sufficient = per_class_df[per_class_df['support'] >= 5]
top10 = sufficient.nlargest(10, 'f1_score')
bottom10 = sufficient.nsmallest(10, 'f1_score')

print("\n🏆 Top 10 Best F1-Scores:")
print(top10[['ingredient', 'f1_score', 'support']].to_string(index=False))

print("\n⚠️  Bottom 10 Worst F1-Scores:")
print(bottom10[['ingredient', 'f1_score', 'support']].to_string(index=False))

# ============================================================================
# Visualizations
# ============================================================================

print("\n" + "="*80)
print("Generating Visualizations")
print("="*80)

# Plot 1: Top 30 F1 Bar Chart
fig, ax = plt.subplots(figsize=(12, 8))
top30 = per_class_df.head(30)
colors = plt.cm.RdYlGn(top30['f1_score'])
ax.barh(range(len(top30)), top30['f1_score'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30['ingredient'], fontsize=10)
ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Top 30 Ingredients - F1 Performance', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
for i, val in enumerate(top30['f1_score']):
    ax.text(val+0.01, i, f'{val:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(EVAL_DIR / 'top30_f1_scores.png', dpi=150, bbox_inches='tight')
print("✓ top30_f1_scores.png")

# Plot 2: Prediction Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pred_counts = preds_binary.sum(axis=1)
actual_counts = all_labels.sum(axis=1)

axes[0].hist(actual_counts, bins=15, color='green', alpha=0.7, edgecolor='black')
axes[0].axvline(actual_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {actual_counts.mean():.1f}')
axes[0].set_title('Ground Truth Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Ingredients per Dish')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(pred_counts, bins=15, color='orange', alpha=0.7, edgecolor='black')
axes[1].axvline(pred_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pred_counts.mean():.1f}')
axes[1].set_title('Prediction Distribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Ingredients per Dish')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(EVAL_DIR / 'distributions.png', dpi=150, bbox_inches='tight')
print("✓ distributions.png")

# Plot 3: Precision-Recall Scatter
fig, ax = plt.subplots(figsize=(11, 8))
top40 = per_class_df[per_class_df['support']>0].head(40)
scatter = ax.scatter(
    top40['recall'], top40['precision'],
    s=top40['support']*12, c=top40['f1_score'],
    cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.8
)
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision vs Recall (Top 40 Ingredients)', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
plt.colorbar(scatter, label='F1-Score')
plt.tight_layout()
plt.savefig(EVAL_DIR / 'precision_recall.png', dpi=150, bbox_inches='tight')
print("✓ precision_recall.png")

# Plot 4: Confidence Distributions
tp_mask = (preds_binary == 1) & (all_labels == 1)
fp_mask = (preds_binary == 1) & (all_labels == 0)
fn_mask = (preds_binary == 0) & (all_labels == 1)

tp_conf = all_preds[tp_mask]
fp_conf = all_preds[fp_mask]
fn_conf = all_preds[fn_mask]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(tp_conf, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2)
axes[0].set_title(f'True Positives ({len(tp_conf):,})', fontweight='bold')
axes[0].set_xlabel('Confidence')

axes[1].hist(fp_conf, bins=30, color='orange', alpha=0.7, edgecolor='black')
axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2)
axes[1].set_title(f'False Positives ({len(fp_conf):,})', fontweight='bold')
axes[1].set_xlabel('Confidence')

axes[2].hist(fn_conf, bins=30, color='red', alpha=0.7, edgecolor='black')
axes[2].axvline(0.5, color='red', linestyle='--', linewidth=2)
axes[2].set_title(f'False Negatives ({len(fn_conf):,})', fontweight='bold')
axes[2].set_xlabel('Confidence')

plt.tight_layout()
plt.savefig(EVAL_DIR / 'confidence_distributions.png', dpi=150, bbox_inches='tight')
print("✓ confidence_distributions.png")

# ============================================================================
# Summary
# ============================================================================

summary = {
    'overall_metrics': overall_metrics,
    'per_class_stats': {
        'total_classes': num_classes,
        'classes_with_support': int((sup > 0).sum()),
        'mean_f1': float(per_class_df['f1_score'].mean())
    },
    'best_ingredients': top10[['ingredient', 'f1_score']].to_dict('records'),
    'worst_ingredients': bottom10[['ingredient', 'f1_score']].to_dict('records')
}

with open(EVAL_DIR / 'evaluation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("Evaluation Complete! 🎉")
print("="*80)
print(f"\n📁 All results saved to: {EVAL_DIR}")
print("\n📊 Generated files:")
print("  ├─ overall_metrics.json")
print("  ├─ per_class_metrics.csv")
print("  ├─ evaluation_summary.json")
print("  ├─ top30_f1_scores.png")
print("  ├─ distributions.png")
print("  ├─ precision_recall.png")
print("  └─ confidence_distributions.png")
