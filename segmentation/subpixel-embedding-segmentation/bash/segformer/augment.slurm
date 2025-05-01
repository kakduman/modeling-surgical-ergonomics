#!/bin/bash
#SBATCH --job-name=augment
#SBATCH --output=logs/augment_%j.out
#SBATCH --error=logs/augment_%j.err

module load Python/3.10.8-GCCcore-12.2.0 
source /home/ksa39/palmer_scratch/surgical-ergonomics/.venv/bin/activate     

cd /home/ksa39/palmer_scratch/surgical-ergonomics/segmentation/subpixel-embedding-segmentation

python external_src/segformer/augment.py \
  --scan-file training/ergonomics/train_scans.txt \
  --mask-file training/ergonomics/train_ground_truths.txt \
  --scan-aug-file training/ergonomics/train_scans_augmented.txt \
  --mask-aug-file training/ergonomics/train_ground_truths_augmented.txt \
  --num-basic 5 \
  --num-new 5
