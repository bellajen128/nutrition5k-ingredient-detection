#!/bin/bash
#SBATCH --job-name=week7_nutrition5k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --output=/scratch/jen.che/nutrition5k_prepared/logs/week7_%j.out
#SBATCH --error=/scratch/jen.che/nutrition5k_prepared/logs/week7_%j.err

echo "============================================================================="
echo "Week 7: Nutrition5K Training Job"
echo "============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Load CUDA
module load cuda/12.3
echo "✓ CUDA loaded"

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Activate virtual environment
source /home/jen.che/nutrition5k_env/bin/activate
echo "✓ Virtual env activated"

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
echo ""

# Run training
cd /home/jen.che/DL_project
echo "Running: python 4_efficientdet_training.py"
echo "============================================================================="
python 4_efficientdet_training.py

echo ""
echo "============================================================================="
echo "Job finished: $(date)"
echo "============================================================================="
