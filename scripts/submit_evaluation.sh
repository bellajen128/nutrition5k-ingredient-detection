#!/bin/bash
#SBATCH --job-name=week8_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem=16GB
#SBATCH --output=/scratch/jen.che/nutrition5k_prepared/logs/eval_%j.out
#SBATCH --error=/scratch/jen.che/nutrition5k_prepared/logs/eval_%j.err

echo "============================================================================="
echo "Week 8: Model Evaluation Job"
echo "============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

module load cuda/12.3
source /home/jen.che/nutrition5k_env/bin/activate

cd /home/jen.che/DL_project
python 5_model_evaluation.py

echo "End: $(date)"
