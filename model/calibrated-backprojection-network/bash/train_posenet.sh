#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# use the kbnet to discover the pose
python src/train_kbnet.py \
--train_image_path training/synchronized/train_image.txt \
--train_sparse_depth_path training/synchronized/train_depth.txt \
--train_intrinsics_path training/synchronized/train_intrinsics.txt \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--min_pool_sizes_sparse_to_dense_pool 15 17 19 \
--max_pool_sizes_sparse_to_dense_pool 23 27 \
--n_convolution_sparse_to_dense_pool 3  \
--n_filter_sparse_to_dense_pool 8 \
--n_filters_encoder_image 48 96 192 384 384 \
--n_filters_encoder_depth 16 32 64 128 128 \
--resolutions_backprojection 0 1 2 3 \
--n_filters_decoder 256 128 128 64 12 \
--deconv_type up \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--learning_rates 1e-4 1e-4  \
--learning_schedule 1000 5000 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_crop_type horizontal vertical anchored \
--augmentation_random_remove_points 0.30 0.60 \
--augmentation_random_noise_type none \
--augmentation_random_noise_spread -1 \
--w_color 0.15 \
--w_structure 0.95 \
--w_sparse_depth 2.00 \
--w_smoothness 2.00 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--n_summary 100 \
--n_checkpoint 1000 \
--checkpoint_path trained_kbnet/void1500/kbnet_model \
--device cpu \
# --device gpu \
--n_thread 8
#--depth_model_restore_path trained_kbnet/void1500/kbnet_model/depth_model-5000.pth \
#--pose_model_restore_path trained_kbnet/void1500/kbnet_model/pose_model-5000.pth \