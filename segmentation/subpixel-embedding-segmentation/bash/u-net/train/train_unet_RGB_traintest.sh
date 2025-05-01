export CUDA_VISIBLE_DEVICES=0

python external_src/generic_unet/train_generic_unet_RGB.py \
--train_image_path training/ergonomics/train_scans.txt \
--train_ground_truth_path training/ergonomics/train_ground_truths.txt \
--n_batch 8 \
--n_chunk 3 \
--n_height 480 \
--n_width 640 \
--dataset_normalization standard \
--dataset_means 30.20063 \
--dataset_stddevs 35.221165 \
--encoder_type_segmentation vggnet13 \
--n_filters_encoder_segmentation 64 128 256 512 512 \
--decoder_type_segmentation generic \
--n_filters_decoder_segmentation 512 256 128 64 \
--weight_initializer kaiming_uniform \
--activation_func leaky_relu \
--use_batch_norm \
--learning_rates 3e-4 \
--learning_schedule 160 \
--positive_class_sample_rates 0.95 \
--positive_class_sample_schedule -1 \
--positive_class_size_thresholds 0 \
--augmentation_probabilities 1.00 \
--augmentation_schedule 160 \
--augmentation_rotate 30 \
--augmentation_crop_and_pad 0.9 1.0 \
--loss_func_segmentation cross_entropy weight_decay \
--w_cross_entropy 1.00 \
--w_weight_decay_segmentation 1e-7 \
--w_positive_class 1.50 \
--n_summary 1 \
--n_checkpoint 200 \
--checkpoint_path \
 trained_models/unet/unet_traintest \
--device gpu \
--n_thread 8
