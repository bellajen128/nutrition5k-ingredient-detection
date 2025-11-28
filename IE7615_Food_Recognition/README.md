# IE7615 Food Ingredient Recognition System

Deep learning-based multi-label food ingredient detection and nutritional analysis.

## Project Overview

This project implements an end-to-end system for:
- Detecting 249 types of food ingredients from images
- Quantifying prediction uncertainty
- Estimating nutritional content
- Providing dietary recommendations

**Model Performance**: F1 Score 0.786 | Precision 0.781 | Recall 0.824

## Repository Structure
```
IE7615_Food_Recognition/
├── training/           # Training scripts (run on HPC)
├── deployment/         # Streamlit app (for deployment)
├── docs/              # Documentation
└── README.md          # This file
```

## Key Features

### 1. EfficientNet-B3 + MFB Class Weighting
- Handles class imbalance using Median Frequency Balancing
- 16.3M parameters
- Trained on 2,792 dishes

### 2. Threshold Optimization
- Data-driven threshold selection
- Optimal threshold: 0.25
- Improves F1 by 0.9%

### 3. MC Dropout Uncertainty Estimation
- 20 forward passes per prediction
- Quantifies prediction confidence
- Identifies uncertain predictions

### 4. Ingredient Co-occurrence Learning
- Learns from 2,792 training samples
- 6,999 unique ingredient pairs
- Boosts recall by 3%

### 5. Nutritional Reasoning
- Manual portion size input
- 555 ingredients in database
- Calculates calories, protein, carbs, fat

## Training (HPC)

### Requirements
- SLURM cluster with GPU
- Python 3.12+
- See training/requirements.txt

### Training Scripts
```bash
cd training/

# 1. Train base model with MFB weighting
sbatch run_training.slurm

# 2. Optimize threshold
python 12_threshold_final.py

# 3. Evaluate uncertainty
python 13_uncertainty_with_optimal_threshold.py

# 4. Test co-occurrence learning
python 19_ingredient_cooccurrence.py
```

## Deployment (Streamlit)

### Local Testing
```bash
cd deployment/
pip install -r requirements.txt
streamlit run app.py
```

### Cloud Deployment

1. Push to GitHub
2. Visit share.streamlit.io
3. Connect repository
4. Deploy deployment/ folder

**Note**: Model file (134MB) may need Git LFS or alternative hosting.

## Dataset

**Nutrition5K** (Google Research)
- 3,490 RGB images
- 3,490 Depth images  
- 249 ingredient classes
- Ground truth labels

**Note**: Original dataset not included due to size. Download from official source.

## Model Architecture
```
Input (512x512 RGB)
    ↓
EfficientNet-B3 Backbone (frozen ImageNet weights)
    ↓
Custom Classifier (2-layer MLP + Dropout)
    ↓
BCEWithLogitsLoss + MFB Class Weights
    ↓
Output (249-dim multi-label)
```

## Results Summary

| Metric | Value |
|--------|-------|
| F1 Score | 0.786 |
| Precision | 0.781 |
| Recall | 0.824 |
| Exact Match | 0.370 |
| Hamming Accuracy | 0.985 |

## Citation
```
IE7615 Deep Learning Project
Northeastern University
November 2025
```

## License

Educational use only.
