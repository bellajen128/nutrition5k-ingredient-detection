"""
Ingredient Co-occurrence Learning (Updated for B3+MFB)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms
from pathlib import Path
import json
from tqdm import tqdm
import timm
from PIL import Image
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

print("="*60)
print("Ingredient Co-occurrence Learning")
print("="*60)

# ============================================================
# 1. Learn Co-occurrence from Training Data
# ============================================================
print("\n1. Learning ingredient co-occurrences from training data...")

TRAIN_CSV = "/scratch/jen.che/nutrition5k_prepared/train.csv"
VOCAB_JSON = "/scratch/jen.che/nutrition5k_prepared/ingredient_vocab.json"

with open(VOCAB_JSON, 'r') as f:
    vocab = json.load(f)['vocab']

train_df = pd.read_csv(TRAIN_CSV)

# Build co-occurrence matrix
num_classes = len(vocab)
cooccurrence_matrix = np.zeros((num_classes, num_classes))
ingredient_counts = np.zeros(num_classes)

print(f"Processing {len(train_df)} training samples...")

for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    ingredients = eval(row['ingredients'])
    
    indices = []
    for ing in ingredients:
        if ing in vocab:
            indices.append(vocab.index(ing))
    
    for i in indices:
        ingredient_counts[i] += 1
    
    for i in indices:
        for j in indices:
            if i != j:
                cooccurrence_matrix[i, j] += 1

# Calculate conditional probability
cooccurrence_prob = np.zeros_like(cooccurrence_matrix)
for i in range(num_classes):
    if ingredient_counts[i] > 0:
        cooccurrence_prob[i, :] = cooccurrence_matrix[i, :] / ingredient_counts[i]

print(f"✅ Co-occurrence matrix computed")
print(f"  Total unique ingredient pairs: {np.sum(cooccurrence_matrix > 0) / 2:.0f}")

# Find top co-occurrences
print("\nTop 10 ingredient pairs:")
pairs = []
for i in range(num_classes):
    for j in range(i+1, num_classes):
        if cooccurrence_matrix[i, j] > 10:
            prob = (cooccurrence_matrix[i, j] / max(ingredient_counts[i], 1))
            pairs.append((vocab[i], vocab[j], cooccurrence_matrix[i, j], prob))

pairs.sort(key=lambda x: x[2], reverse=True)
for ing1, ing2, count, prob in pairs[:10]:
    print(f"  {ing1:20s} + {ing2:20s}: {count:3.0f} times ({prob:.2%})")

# ============================================================
# 2. Load Model (B3 Architecture)
# ============================================================
print("\n2. Loading model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FoodModel(nn.Module):
    """Match training architecture exactly"""
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=False,
            num_classes=0,
            global_pool=''
        )
        
        self.feature_dim = 1536
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

MODEL_PATH = "/scratch/jen.che/nutrition5k_prepared/efficientnet_b3_weighted_outputs/models/efficientnet_best.pth"
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

model = FoodModel(len(vocab), dropout=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("✅ Model loaded")

# ============================================================
# 3. Prediction with Co-occurrence Boosting
# ============================================================
def predict_with_cooccurrence(model, image_tensor, cooccurrence_prob, 
                               base_threshold=0.25, boost_factor=0.15):
    """Predict with co-occurrence boosting"""
    with torch.no_grad():
        output = model(image_tensor)
        base_probs = torch.sigmoid(output).cpu().numpy()[0]
    
    boosted_probs = base_probs.copy()
    detected = np.where(base_probs > base_threshold)[0]
    
    for ingredient_idx in detected:
        related_probs = cooccurrence_prob[ingredient_idx]
        
        for j in range(len(vocab)):
            if j != ingredient_idx and related_probs[j] > 0.3:
                boost = related_probs[j] * boost_factor
                boosted_probs[j] = min(1.0, boosted_probs[j] + boost)
    
    return base_probs, boosted_probs, detected

# ============================================================
# 4. Test on Validation Set
# ============================================================
print("\n3. Testing with co-occurrence boosting...\n")

VAL_CSV = "/scratch/jen.che/nutrition5k_prepared/val.csv"
IMG_DIR = "/scratch/jen.che/nutrition5k_prepared/images"
val_df = pd.read_csv(VAL_CSV)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

BASE_THRESHOLD = 0.25

all_labels = []
all_base_preds = []
all_boosted_preds = []

print(f"Processing {len(val_df)} validation samples...")

for idx in tqdm(range(len(val_df))):
    row = val_df.iloc[idx]
    dish_id = row['dish_id']
    
    img_path = Path(IMG_DIR) / f"{dish_id}.jpg"
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    base_probs, boosted_probs, detected = predict_with_cooccurrence(
        model, image_tensor, cooccurrence_prob, BASE_THRESHOLD
    )
    
    ingredients = eval(row['ingredients'])
    labels = np.zeros(num_classes)
    for ing in ingredients:
        if ing in vocab:
            labels[vocab.index(ing)] = 1.0
    
    all_labels.append(labels)
    all_base_preds.append((base_probs > BASE_THRESHOLD).astype(int))
    all_boosted_preds.append((boosted_probs > BASE_THRESHOLD).astype(int))

all_labels = np.array(all_labels)
all_base_preds = np.array(all_base_preds)
all_boosted_preds = np.array(all_boosted_preds)

# ============================================================
# 5. Evaluate Performance
# ============================================================
print("\n" + "="*60)
print("Performance Comparison")
print("="*60)

base_precision = precision_score(all_labels, all_base_preds, average='samples', zero_division=0)
base_recall = recall_score(all_labels, all_base_preds, average='samples', zero_division=0)
base_f1 = f1_score(all_labels, all_base_preds, average='samples', zero_division=0)

print(f"\nBase Model (threshold={BASE_THRESHOLD}):")
print(f"  Precision: {base_precision:.3f}")
print(f"  Recall:    {base_recall:.3f}")
print(f"  F1 Score:  {base_f1:.3f}")
print(f"  Avg preds: {all_base_preds.sum(axis=1).mean():.1f}")

boosted_precision = precision_score(all_labels, all_boosted_preds, average='samples', zero_division=0)
boosted_recall = recall_score(all_labels, all_boosted_preds, average='samples', zero_division=0)
boosted_f1 = f1_score(all_labels, all_boosted_preds, average='samples', zero_division=0)

print(f"\nWith Co-occurrence Boosting:")
print(f"  Precision: {boosted_precision:.3f}")
print(f"  Recall:    {boosted_recall:.3f}")
print(f"  F1 Score:  {boosted_f1:.3f}")
print(f"  Avg preds: {all_boosted_preds.sum(axis=1).mean():.1f}")

f1_improvement = boosted_f1 - base_f1
recall_improvement = boosted_recall - base_recall

print(f"\n" + "="*60)
print("Improvement")
print("="*60)
print(f"  F1 Score:  {f1_improvement:+.4f} ({(f1_improvement/base_f1*100):+.2f}%)")
print(f"  Recall:    {recall_improvement:+.4f} ({(recall_improvement/base_recall*100):+.2f}%)")

if f1_improvement > 0:
    print(f"\n✅ Co-occurrence learning improves performance!")
else:
    print(f"\n→ Base model performs similarly or better")

# ============================================================
# 6. Show Examples
# ============================================================
print(f"\n" + "="*60)
print("Example Predictions")
print("="*60)

for idx in [0, 50, 100]:
    if idx >= len(val_df):
        continue
        
    row = val_df.iloc[idx]
    dish_id = row['dish_id']
    
    true_ings = eval(row['ingredients'])
    base_pred_idx = np.where(all_base_preds[idx] == 1)[0]
    boosted_pred_idx = np.where(all_boosted_preds[idx] == 1)[0]
    
    print(f"\nDish: {dish_id}")
    print(f"True ({len(true_ings)}): {', '.join(true_ings[:5])}...")
    print(f"Base ({len(base_pred_idx)}): {', '.join([vocab[i] for i in base_pred_idx[:5]])}...")
    print(f"Boosted ({len(boosted_pred_idx)}): {', '.join([vocab[i] for i in boosted_pred_idx[:5]])}...")
    
    added = set(boosted_pred_idx) - set(base_pred_idx)
    if added:
        print(f"  Added: {', '.join([vocab[i] for i in added])}")

# ============================================================
# 7. Save Results
# ============================================================
output_dir = Path('/scratch/jen.che/nutrition5k_prepared/cooccurrence_outputs')
output_dir.mkdir(exist_ok=True)

np.save(output_dir / 'cooccurrence_matrix.npy', cooccurrence_matrix)
np.save(output_dir / 'cooccurrence_prob.npy', cooccurrence_prob)

results = {
    'model': 'EfficientNet-B3 + MFB',
    'threshold': BASE_THRESHOLD,
    'base': {
        'precision': float(base_precision),
        'recall': float(base_recall),
        'f1': float(base_f1)
    },
    'boosted': {
        'precision': float(boosted_precision),
        'recall': float(boosted_recall),
        'f1': float(boosted_f1)
    },
    'improvement': {
        'f1': float(f1_improvement),
        'recall': float(recall_improvement)
    }
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_dir}")
print("\nIngredient Co-occurrence Learning Complete!")
