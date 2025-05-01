# module load Python/3.10.8-GCCcore-12.2.0 
# source /home/ksa39/palmer_scratch/surgical-ergonomics/.venv/bin/activate     

# cd /home/ksa39/palmer_scratch/surgical-ergonomics/segmentation/subpixel-embedding-segmentation

python external_src/segformer/train_segformer.py \
  --scan-aug-file training/ergonomics/train_scans_augmented.txt \
  --mask-aug-file training/ergonomics/train_ground_truths_augmented.txt \
  --val-size 99 \
  --model-checkpoint nvidia/mit-b4 \
  --batch-size 8 \
  --num-epochs 2
