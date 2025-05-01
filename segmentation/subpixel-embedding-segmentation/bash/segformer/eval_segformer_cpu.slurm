#!/bin/bash
#SBATCH --job-name=eval_segformer_cpu
#SBATCH --output=logs/eval_segformer_cpu_%j.out
#SBATCH --error=logs/eval_segformer_cpu_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G

module load Python/3.10.8-GCCcore-12.2.0 
source /home/ksa39/palmer_scratch/surgical-ergonomics/.venv/bin/activate     

cd /home/ksa39/palmer_scratch/surgical-ergonomics/segmentation/subpixel-embedding-segmentation

python external_src/segformer/evaluate_segformer.py