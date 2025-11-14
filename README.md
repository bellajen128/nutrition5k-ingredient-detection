# Nutrition5K Ingredient Detection

**Authors**: Chen Jen, Qingyi Ji  
**Course**: Deep Learning (Northeastern University)  
**Date**: Fall 2025

---

## Project Overview

An automated food ingredient detection system using deep learning to enable smart dietary logging. The system analyzes food images and identifies individual ingredients using multi-label classification.

### Key Features
- Multi-ingredient detection from a single food image
- 249 ingredient categories
- 75.73% F1-Score on test set
- Fast inference (~50ms per image)

---

## Repository Structure

```
nutrition5k-ingredient-detection/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup_notebook_and_training_env.py  # Environment setup script
│
├── notebooks/                          # Jupyter notebooks
│   ├── 1_EDA.ipynb                    # Exploratory data analysis
│   ├── 2_Preprocessing.ipynb          # Data preparation
│   └── 3_training.ipynb               # Initial experiments
│
├── scripts/                            # Training & evaluation scripts
│   ├── 4_efficientdet_training.py     # Main training script
│   ├── 5_model_evaluation.py          # Model evaluation
│   ├── submit_week7_training.sh       # SLURM job for training
│   └── submit_evaluation.sh           # SLURM job for evaluation
│
├── results/                            # Training results and figures
│   ├── training_summary.json
│   ├── training_history.csv
│   ├── training_curves.png
│   └── overall_metrics.json
│
└── docs/                               # Documentation
    └── progress_report.pdf
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA 12.1+ (for GPU training)
- 16GB+ RAM
- GPU with 32GB VRAM (recommended: V100 or better)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nutrition5k-ingredient-detection.git
cd nutrition5k-ingredient-detection

# Create virtual environment
python -m venv nutrition5k_env
source nutrition5k_env/bin/activate  # Linux/Mac
# or
nutrition5k_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

**Download Nutrition5K Dataset:**
1. Visit: https://www.kaggle.com/datasets/siddhantrout/nutrition5k-dataset
2. Download `archive.zip`
3. Extract to `./data/nutrition5k/`

**Preprocess data:**
```bash
jupyter notebook notebooks/2_Preprocessing.ipynb
# Run all cells to generate:
# - train.csv, val.csv, test.csv
# - ingredient_vocab.json
# - images/ directory
```

---

## How to Run

### 1. Exploratory Data Analysis

```bash
jupyter notebook notebooks/1_EDA.ipynb
```

**Output:**
- Dataset statistics
- Ingredient frequency distribution
- Nutrition value analysis

### 2. Data Preprocessing

```bash
jupyter notebook notebooks/2_Preprocessing.ipynb
```

**Output:**
- Extracted images: `data/images/*.jpg`
- Data splits: `train.csv`, `val.csv`, `test.csv`
- Vocabulary: `ingredient_vocab.json`

### 3. Model Training

**Local machine:**
```bash
python scripts/4_efficientdet_training.py
```

**HPC (SLURM):**
```bash
sbatch scripts/submit_week7_training.sh
```

**Output:**
- Best model: `models/efficientnet_best.pth`
- Training history: `logs/training_history.csv`
- Visualizations: `visualizations/training_curves.png`

### 4. Model Evaluation

```bash
python scripts/5_model_evaluation.py
```

**Output:**
- Test metrics: `evaluation/overall_metrics.json`
- Per-class analysis: `evaluation/per_class_metrics.csv`
- Visualization plots in `evaluation/`

---

## Dependencies

### Core Libraries

```
torch>=2.0.0
torchvision>=0.15.0
timm==0.9.16              # EfficientNet models
effdet==0.4.1             # EfficientDet utilities
```

### Data Processing

```
pandas>=2.0.0
numpy>=1.24.0
openpyxl                  # Excel file reading
Pillow>=9.0.0             # Image processing
opencv-python>=4.8.0
```

### Evaluation & Visualization

```
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm                      # Progress bars
```

### Optional (for deployment)

```
streamlit>=1.28.0         # Web interface
gradio>=3.50.0            # Alternative UI
```

**Full list:** See `requirements.txt`

---

## Model Architecture

### EfficientNet-B0 Multi-Label Classifier

```
Input: 512×512 RGB Image
  ↓
[EfficientNet-B0 Backbone]
  - Pretrained on ImageNet
  - Extracts 1280-dim features
  ↓
[Classification Head]
  - Global Average Pooling
  - FC: 1280 → 512 (+ ReLU + Dropout 0.3)
  - FC: 512 → 249 (ingredient logits)
  ↓
[Sigmoid Activation]
  ↓
Output: 249 probabilities [0, 1]
  - Threshold: 0.5 for binary prediction
```

**Parameters:** 4.2M trainable

---

## Results

### Model Performance

| Metric | Validation | Test | Target | Status |
|--------|-----------|------|--------|--------|
| **F1-Score** | 77.70% | **75.73%** | >70% | Done |
| **Precision** | 82.54% | **83.92%** | >70% | Done |
| **Recall** | 75.62% | **72.34%** | >60% | Done |

### Training Efficiency

- **Training Time:** 8.4 minutes (20 epochs)
- **Hardware:** NVIDIA V100 (32GB)
- **Best Epoch:** 18
- **Convergence:** Stable, no overfitting

### Key Insights

Strengths:
- High precision (83.92%) - Low false positive rate
- Fast training - Efficient for iteration
- Good generalization - Test ≈ Validation

Limitations:
- Recall at 72% - ~28% ingredients missed
- Rare ingredients perform poorly (low support)
- Dataset bias toward Western cuisine

---

## Scripts Explanation

### `setup_notebook_and_training_env.py`
**Purpose:** Initial environment setup  
**Usage:** `python setup_notebook_and_training_env.py`  
**Output:** Creates virtual environment and downloads dataset

### `notebooks/1_EDA.ipynb`
**Purpose:** Exploratory Data Analysis  
**Key Functions:**
- Load Nutrition5K dataset
- Compute statistics (ingredient frequency, nutrition distribution)
- Generate visualizations

### `notebooks/2_Preprocessing.ipynb`
**Purpose:** Data preparation for training  
**Key Steps:**
1. Decode images from pickle files
2. Build ingredient vocabulary (249 classes)
3. Create multi-hot label vectors
4. Split into train/val/test (80/10/10)

### `scripts/4_efficientdet_training.py`
**Purpose:** Train EfficientNet-B0 for ingredient detection  
**Key Components:**
- **Dataset class:** `Nutrition5KDataset` (multi-label)
- **Model:** `EfficientNetMultiLabel` (custom head)
- **Training loop:** 20 epochs with validation
- **Optimization:** AdamW + Cosine Annealing LR
- **Checkpointing:** Saves best model based on val F1

**Usage:**
```bash
# On HPC
sbatch scripts/submit_week7_training.sh

# Local (with GPU)
python scripts/4_efficientdet_training.py
```

### `scripts/5_model_evaluation.py`
**Purpose:** Detailed evaluation on test set  
**Metrics Computed:**
- Overall: Precision, Recall, F1, Exact Match, Hamming Accuracy
- Per-class: Individual ingredient performance
- Error analysis: Worst predictions, confusion cases

**Usage:**
```bash
python scripts/5_model_evaluation.py
```

---

## Dataset Access

### Nutrition5K Dataset

**Source:** [Kaggle - Nutrition5K](https://www.kaggle.com/datasets/siddhantrout/nutrition5k-dataset)

**Contents:**
- `dish_ingredients.xlsx`: Ingredient lists for 5,006 dishes
- `dishes.xlsx`: Nutritional summaries
- `ingredients.xlsx`: Per-ingredient nutrition values
- `dish_images.pkl`: RGB and depth images

**Citation:**
```
Thames, Q., Karpur, A., Norris, W., Xia, F., Panait, L., Weyand, T., & Sim, J. (2021).
Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food.
arXiv:2103.03375
```

**Setup Instructions:**
1. Download dataset from Kaggle
2. Place `archive.zip` in `data/nutrition5k/`
3. Run preprocessing notebook to extract and prepare data

---

## Training on HPC

### Northeastern Discovery Cluster

**Environment:**
- OS: Rocky Linux 9
- Scheduler: SLURM
- GPUs: V100-SXM2 (32GB)
- CUDA: 12.3

**Setup:**
```bash
# Load CUDA module
module load cuda/12.3

# Activate virtual environment
source ~/nutrition5k_env/bin/activate

# Submit training job
sbatch scripts/submit_week7_training.sh
```

**Monitor Progress:**
```bash
# Check job status
squeue -u $USER

# View output
tail -f logs/week7_*.out
```

---

## Training Configuration

```yaml
Model:
  Architecture: EfficientNet-B0
  Pretrained: ImageNet
  Parameters: 4.2M trainable

Data:
  Image size: 512×512
  Augmentation:
    - Random horizontal flip (p=0.5)
    - Random rotation (±10°)
    - Color jitter (brightness/contrast/saturation ±0.2)

Training:
  Epochs: 20
  Batch size: 32
  Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
  Scheduler: CosineAnnealingLR
  Loss: BCEWithLogitsLoss

Hardware:
  GPU: V100-SXM2 (32GB)
  Time: 8.4 minutes
```

---

## Future Work

### Week 9-10: Prediction Visualization & Demo
- Build prediction visualization tool
- Create Streamlit web interface
- Implement portion size input and nutrition calculation

### Week 11-13: Testing & Refinement
- Threshold optimization
- Error analysis and model improvement
- Cross-validation experiments

### Week 14-16: Final Deliverables
- Complete technical report
- Demo video
- Presentation slides

---

## Contact

**Chen Jen** - jen.che@northeastern.edu  
**Qingyi Ji** - ji.qin@northeastern.edu

---

## Acknowledgments

- **Dataset:** Google Research - Nutrition5K
- **Framework:** PyTorch, timm (Ross Wightman)
- **Infrastructure:** Northeastern Discovery HPC Cluster
