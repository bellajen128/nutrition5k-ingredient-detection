# Training Scripts

Scripts for training on HPC with SLURM.

## Prerequisites

- SLURM cluster with GPU (V100 recommended)
- Python 3.12+
- Virtual environment with PyTorch, timm, etc.

## Dataset Setup

Place Nutrition5K dataset at:
```
/scratch/username/nutrition5k_prepared/
├── train.csv
├── val.csv
├── test.csv
├── images/
├── depth_images/
└── ingredient_vocab.json
```

## Training Pipeline

### 1. Train Base Model (EfficientNet-B3 + MFB)
```bash
sbatch run_training.slurm  # Or modify paths in script
```

Output: `efficientnet_best.pth` with F1 ~0.78

### 2. Optimize Threshold
```bash
python 12_threshold_final.py
```

Finds optimal threshold (expected: 0.25)

### 3. Uncertainty Estimation
```bash
python 13_uncertainty_with_optimal_threshold.py
```

MC Dropout with 20 samples, saves uncertainty scores.

### 4. Co-occurrence Learning
```bash
python 19_ingredient_cooccurrence.py
```

Tests co-occurrence boosting effect.

## Scripts

- `5_efficientnet_weighted_training.py` - Main training with MFB weighting
- `12_threshold_final.py` - Threshold optimization
- `13_uncertainty_with_optimal_threshold.py` - MC Dropout uncertainty
- `19_ingredient_cooccurrence.py` - Co-occurrence learning evaluation

## Expected Outputs

After training:
```
/scratch/username/nutrition5k_prepared/
├── efficientnet_b3_weighted_outputs/
│   └── models/efficientnet_best.pth
├── threshold_optimization/
│   └── results.csv
├── uncertainty_outputs/
│   └── uncertainty_config.json
└── cooccurrence_outputs/
    └── cooccurrence_prob.npy
```
