export CUDA_VISIBLE_DEVICES=0

python external_src/generic_unet/run_generic_unet.py \
--multimodal_scan_paths testing/ergonomics/test_scans.txt \
--n_chunk 3 \
--dataset_normalization standard \
--dataset_means 30.20063 \
--dataset_stddevs 35.221165 \
--encoder_type_segmentation vggnet13 \
--n_filters_encoder_segmentation 64 128 256 512 512 \
--decoder_type_segmentation generic \
--n_filters_decoder_segmentation 512 256 128 64 \
--augmentation_flip_type none \
--activation_func leaky_relu \
--use_batch_norm \
--do_visualize_predictions \
--device gpu \
--checkpoint_path evaluation_results/u-net/unet_traintest \
--restore_path \
trained_models/unet/unet_traintest/model-600.pth \
--n_thread 8
