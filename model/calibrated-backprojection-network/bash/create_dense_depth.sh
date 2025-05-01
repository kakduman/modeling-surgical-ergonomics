# module load Python/3.10.8-GCCcore-12.2.0 
# source /home/ksa39/palmer_scratch/surgical-ergonomics/.venv/bin/activate     

# cd /home/ksa39/palmer_scratch/surgical-ergonomics/model/calibrated-backprojection-network
python setup/create_depthany.py \
    --encoder vitl \
    --weights_path ../../Depth-Anything-V2-main/depth_anything_v2/depth_anything_v2_vitl.pth \
    --data3d_root ../../data/data-3d/ \