"""
Uncertainty Estimation with Optimal Threshold (0.25)
Updated for EfficientNet-B3 + MFB
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

print("="*60)
print("Uncertainty Estimation with Optimal Threshold")
print("="*60)

# Load vocab
VOCAB_JSON = "/scratch/jen.che/nutrition5k_prepared/ingredient_vocab.json"
with open(VOCAB_JSON, 'r') as f:
    vocab_data = json.load(f)
    vocab = vocab_data['vocab']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load model
MODEL_PATH = "/scratch/jen.che/nutrition5k_prepared/efficientnet_b3_weighted_outputs/models/efficientnet_best.pth"
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

# ============================================================
# CORRECTED MODEL ARCHITECTURE (Match training script exactly!)
# ============================================================
class ModelWithMCDropout(nn.Module):
    """
    Must match EXACTLY the architecture in 5_efficientnet_weighted_training.py
    """
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        
        # Backbone - match training script
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=False,
            num_classes=0,
            global_pool=''  # Important! Match training
        )
        
        self.feature_dim = 1536
        
        # Classifier - match training script EXACTLY
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
    
    def predict_with_uncertainty(self, x, n_samples=20):
        """MC Dropout prediction"""
        # Enable dropout during inference
        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu().numpy())
        
        self.eval()
        
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std

model = ModelWithMCDropout(len(vocab), dropout=0.3)

try:
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model loaded successfully")
except RuntimeError as e:
    print("❌ Error loading model!")
    print(f"Error: {e}")
    exit(1)

model = model.to(device)
model.eval()

print(f"Model epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Model val F1: {checkpoint.get('metrics', {}).get('f1', 'unknown'):.4f}")
print()

OPTIMAL_THRESHOLD = 0.25
print(f"Using optimal threshold: {OPTIMAL_THRESHOLD}\n")

TEST_CSV = "/scratch/jen.che/nutrition5k_prepared/test.csv"
IMG_DIR = "/scratch/jen.che/nutrition5k_prepared/images"
test_df = pd.read_csv(TEST_CSV)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("="*60)
print(f"Testing on {min(20, len(test_df))} samples")
print("="*60)

results = []
n_test = min(20, len(test_df))

for idx in tqdm(range(n_test)):
    row = test_df.iloc[idx]
    dish_id = row['dish_id']
    
    img_path = Path(IMG_DIR) / f"{dish_id}.jpg"
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    mean_pred, std_pred = model.predict_with_uncertainty(image_tensor, n_samples=20)
    
    mean_pred = mean_pred[0]
    std_pred = std_pred[0]
    
    predicted_idx = np.where(mean_pred > OPTIMAL_THRESHOLD)[0]
    
    true_ingredients = eval(row['ingredients'])
    
    true_set = set(true_ingredients)
    pred_set = set([vocab[i] for i in predicted_idx])
    
    correct = len(true_set & pred_set)
    precision = correct / len(pred_set) if len(pred_set) > 0 else 0
    recall = correct / len(true_set) if len(true_set) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Sample {idx+1}: {dish_id}")
    print(f"{'='*60}")
    print(f"True: {len(true_ingredients)} | Predicted: {len(predicted_idx)} | Correct: {correct}")
    print(f"Precision: {precision:.2f} | Recall: {recall:.2f}")
    
    if len(predicted_idx) > 0:
        print(f"\nTop 10 predictions with uncertainty:")
        
        sorted_idx = predicted_idx[np.argsort(mean_pred[predicted_idx])[::-1]][:10]
        
        for i, pred_idx in enumerate(sorted_idx):
            ingredient = vocab[pred_idx]
            conf = mean_pred[pred_idx]
            unc = std_pred[pred_idx]
            
            if unc < 0.05:
                level = "Very Confident ✓✓✓"
            elif unc < 0.10:
                level = "Confident ✓✓"
            elif unc < 0.20:
                level = "Uncertain ?"
            else:
                level = "Very Uncertain ??"
            
            correct_mark = "✓" if ingredient in true_ingredients else "✗"
            
            print(f"  {i+1:2d}. {ingredient:20s} {conf:.3f}±{unc:.3f} [{level}] {correct_mark}")
    
    results.append({
        'dish_id': dish_id,
        'num_true': len(true_ingredients),
        'num_predicted': len(predicted_idx),
        'num_correct': correct,
        'precision': precision,
        'recall': recall,
        'mean_uncertainty': std_pred[predicted_idx].mean() if len(predicted_idx) > 0 else 0,
        'max_uncertainty': std_pred[predicted_idx].max() if len(predicted_idx) > 0 else 0,
    })

results_df = pd.DataFrame(results)
output_dir = Path('/scratch/jen.che/nutrition5k_prepared/uncertainty_outputs')
output_dir.mkdir(exist_ok=True)

results_df.to_csv(output_dir / 'uncertainty_with_optimal_threshold.csv', index=False)

print(f"\n{'='*60}")
print("Summary Statistics")
print(f"{'='*60}")
print(f"Avg true ingredients:      {results_df['num_true'].mean():.1f}")
print(f"Avg predicted ingredients: {results_df['num_predicted'].mean():.1f}")
print(f"Avg correct predictions:   {results_df['num_correct'].mean():.1f}")
print(f"Avg precision:             {results_df['precision'].mean():.3f}")
print(f"Avg recall:                {results_df['recall'].mean():.3f}")
print(f"Mean uncertainty:          {results_df['mean_uncertainty'].mean():.4f}")
print(f"Max uncertainty:           {results_df['max_uncertainty'].mean():.4f}")

config = {
    'model': 'EfficientNet-B3 + MFB',
    'model_path': MODEL_PATH,
    'optimal_threshold': OPTIMAL_THRESHOLD,
    'mc_dropout_samples': 20,
    'dropout_rate': 0.3,
    'avg_predictions': float(results_df['num_predicted'].mean()),
    'avg_precision': float(results_df['precision'].mean()),
    'avg_recall': float(results_df['recall'].mean()),
    'mean_uncertainty': float(results_df['mean_uncertainty'].mean())
}

with open(output_dir / 'uncertainty_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Results saved to: {output_dir}")
print("\nUncertainty Estimation Complete!")
