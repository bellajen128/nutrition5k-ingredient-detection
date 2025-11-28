"""
Threshold Optimization - FINAL correct version
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import timm
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

print("="*60)
print("Threshold Optimization - FINAL")
print("="*60)

VOCAB_JSON = "/scratch/jen.che/nutrition5k_prepared/ingredient_vocab.json"
with open(VOCAB_JSON, 'r') as f:
    vocab_data = json.load(f)
    vocab = vocab_data['vocab']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "/scratch/jen.che/nutrition5k_prepared/efficientdet_outputs/models/efficientnet_best.pth"
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

print(f"Stored F1: {checkpoint['metrics']['f1']:.3f}\n")

# Correct architecture: only Dropout and ReLU (no BatchNorm!)
# 0: Dropout
# 1: ReLU (or similar)
# 2: Dropout (or similar)
# 3: Linear(1280, 512)
# 4: Dropout
# 5: ReLU
# 6: Linear(512, 249)

class FinalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),           # 0
            nn.ReLU(),                 # 1
            nn.Dropout(0.3),           # 2
            nn.Linear(1280, 512),      # 3
            nn.Dropout(0.3),           # 4
            nn.ReLU(),                 # 5
            nn.Linear(512, len(vocab)) # 6
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

model = FinalModel(len(vocab))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("✓ Model loaded successfully\n")

# Get predictions
VAL_CSV = "/scratch/jen.che/nutrition5k_prepared/val.csv"
IMG_DIR = "/scratch/jen.che/nutrition5k_prepared/images"
val_df = pd.read_csv(VAL_CSV)

IMG_SIZE = 512

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"Predicting {len(val_df)} samples...")

all_probs = []
all_labels = []

with torch.no_grad():
    for idx in tqdm(range(len(val_df))):
        row = val_df.iloc[idx]
        dish_id = row['dish_id']
        
        img_path = Path(IMG_DIR) / f"{dish_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        output = model(image_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]
        
        ingredients = eval(row['ingredients'])
        labels = np.zeros(len(vocab))
        for ingredient in ingredients:
            if ingredient in vocab:
                labels[vocab.index(ingredient)] = 1.0
        
        all_probs.append(probs)
        all_labels.append(labels)

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Sanity check
sanity_f1 = f1_score(all_labels, (all_probs > 0.5).astype(int), average='samples', zero_division=0)
print(f"\n✓ SANITY CHECK (t=0.5): F1 = {sanity_f1:.3f}\n")

if sanity_f1 < 0.6:
    print("⚠️  F1 is lower than expected! Model may not be loaded correctly.")
    print("Continuing anyway...\n")

# Test thresholds
print("="*60)
print("Testing Thresholds")
print("="*60)

thresholds = np.arange(0.1, 0.9, 0.05)
results = []

for threshold in thresholds:
    preds = (all_probs > threshold).astype(int)
    
    p = precision_score(all_labels, preds, average='samples', zero_division=0)
    r = recall_score(all_labels, preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, preds, average='samples', zero_division=0)
    avg_preds = preds.sum(axis=1).mean()
    
    results.append({'threshold': threshold, 'precision': p, 'recall': r, 'f1': f1, 'avg_preds': avg_preds})
    print(f"t={threshold:.2f}: P={p:.3f} R={r:.3f} F1={f1:.3f} | Avg={avg_preds:.1f}")

results_df = pd.DataFrame(results)
best_idx = results_df['f1'].idxmax()
best = results_df.iloc[best_idx]

print(f"\n{'='*60}")
print(f"BEST THRESHOLD: {best['threshold']:.2f}")
print(f"{'='*60}")
print(f"  F1: {best['f1']:.3f}")
print(f"  Precision: {best['precision']:.3f}")
print(f"  Recall: {best['recall']:.3f}")
print(f"  Avg predictions: {best['avg_preds']:.1f}")

# Save
output_dir = Path('/scratch/jen.che/nutrition5k_prepared/threshold_optimization')
output_dir.mkdir(exist_ok=True)
results_df.to_csv(output_dir / 'results.csv', index=False)

with open(output_dir / 'best_threshold.txt', 'w') as f:
    f.write(f"{best['threshold']:.2f}\n")

print("\n✓ Complete!")
