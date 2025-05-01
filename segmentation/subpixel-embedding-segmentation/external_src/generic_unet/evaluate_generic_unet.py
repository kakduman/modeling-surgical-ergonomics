import os
import random

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# — adjust these paths to your repo layout —
from generic_unet_global_constants import (
    WEIGHT_INITIALIZER,
    ATLAS_MEAN,
    ATLAS_SD
)

ENCODER_TYPE_SEGMENTATION = 'vggnet13'
N_FILTERS_ENCODER_SEGMENTATION = [64, 128, 256, 512, 512]
DECODER_TYPE_SEGMENTATION = 'generic'
N_FILTERS_DECODER_SEGMENTATION = [512, 256, 128, 64]
ACTIVATION_FUNC = 'leaky_relu'
USE_BATCH_NORM   = True
from generic_unet_RGB_model import GenericUNetRGBModel

from torchvision import transforms as T

to_tensor = T.ToTensor()
normalize = T.Normalize(
    # note: if your GenericUNet expects 1-channel, use [ATLAS_MEAN/255.] * 1,
    # but if it’s RGB you’d supply a 3-tuple
    mean=[ATLAS_MEAN/255.0],  
    std=[ATLAS_SD/255.0]
)


class Transforms(object):

    def __init__(self,
                 normalized_image_range=[0, 255],
                 random_flip_type=['none'],
                 random_remove_points=[0.70, 0.70],
                 random_noise_type=['none'],
                 random_noise_spread=-1):
        '''
        Transforms and augmentation class

        Arg(s):
            normalized_image_range : list[float]
                intensity range after normalizing images
            random_flip_type : list[str]
                none, horizontal, vertical
            random_remove_points : list[float]
                percentage of points to remove in range map
            random_noise_type : str
                type of noise to add: gaussian, uniform
            random_noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
        '''

        # Image normalization
        self.normalized_image_range = normalized_image_range

        # Geometric augmentations
        self.do_random_horizontal_flip = True if 'horizontal' in random_flip_type else False
        self.do_random_vertical_flip = True if 'vertical' in random_flip_type else False

        self.do_random_remove_points = True if -1 not in random_remove_points else False
        self.remove_points_range = random_remove_points

        self.do_random_noise = \
            True if (random_noise_type != 'none' and random_noise_spread > 0) else False

        self.random_noise_type = random_noise_type
        self.random_noise_spread = random_noise_spread

    def transform(self,
                  images_arr,
                  range_maps_arr=[],
                  validity_maps_arr=[],
                  random_transform_probability=0.50):
        '''
        Applies transform to images and ground truth

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            range_maps_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            validity_maps_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            random_transform_probability : float
                probability to perform transform
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
            list[torch.Tensor[float32]] : list of transformed N x c x H x W range maps tensors
        '''

        device = images_arr[0].device

        n_dim = images_arr[0].ndim

        if n_dim == 4:
            n_batch, _, n_height, n_width = images_arr[0].shape
        elif n_dim == 5:
            n_batch, _, _, n_height, n_width = images_arr[0].shape
        else:
            raise ValueError('Unsupported number of dimensions: {}'.format(n_dim))

        do_random_transform = \
            np.random.rand(n_batch) <= random_transform_probability

        # Normalize images to a given range
        images_arr = self.normalize_images(
            images_arr,
            normalized_image_range=self.normalized_image_range)

        if self.do_random_horizontal_flip:

            do_horizontal_flip = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            images_arr = self.horizontal_flip(
                images_arr,
                do_horizontal_flip)

            range_maps_arr = self.horizontal_flip(
                range_maps_arr,
                do_horizontal_flip)

            validity_maps_arr = self.horizontal_flip(
                validity_maps_arr,
                do_horizontal_flip)

        if self.do_random_vertical_flip:

            do_vertical_flip = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            images_arr = self.vertical_flip(
                images_arr,
                do_vertical_flip)

            range_maps_arr = self.vertical_flip(
                range_maps_arr,
                do_vertical_flip)

            validity_maps_arr = self.vertical_flip(
                validity_maps_arr,
                do_vertical_flip)

        if self.do_random_remove_points:

            do_remove_points = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            values = torch.rand(n_batch, device=device)

            remove_points_min, remove_points_max = self.remove_points_range

            densities = \
                (remove_points_max - remove_points_min) * values + remove_points_min

            range_maps_arr = self.remove_random_nonzero(
                images_arr=range_maps_arr,
                do_remove=do_remove_points,
                densities=densities)

        if self.do_random_noise:

            do_add_noise = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            range_maps_arr = self.add_noise(
                range_maps_arr,
                do_add_noise=do_add_noise,
                noise_type=self.random_noise_type,
                noise_spread=self.random_noise_spread)

        # Return the transformed inputs
        outputs = []

        if len(images_arr) > 0:
            outputs.append(images_arr)

        if len(range_maps_arr) > 0:
            outputs.append(range_maps_arr)

        if len(validity_maps_arr) > 0:
            outputs.append(validity_maps_arr)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    '''
    Photometric transforms
    '''
    def normalize_images(self, images_arr, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            images_arr[torch.Tensor[float32]] : list of normalized N x C x H x W tensors
        '''

        if normalized_image_range == [0, 1]:
            images_arr = [
                images / 255.0 for images in images_arr
            ]
        elif normalized_image_range == [-1, 1]:
            images_arr = [
                2.0 * (images / 255.0) - 1.0 for images in images_arr
            ]
        elif normalized_image_range == [0, 255]:
            pass
        else:
            raise ValueError('Unsupported normalization range: {}'.format(
                normalized_image_range))

        return images_arr

    '''
    Geometric transforms
    '''
    def horizontal_flip(self, images_arr, do_horizontal_flip):
        '''
        Perform horizontal flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_horizontal_flip : bool
                N booleans to determine if horizontal flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_horizontal_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-1])

            images_arr[i] = images

        return images_arr

    def vertical_flip(self, images_arr, do_vertical_flip):
        '''
        Perform vertical flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_vertical_flip : bool
                N booleans to determine if vertical flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_vertical_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-2])

            images_arr[i] = images

        return images_arr

    def remove_random_nonzero(self, images_arr, do_remove, densities):
        '''
        Remove random nonzero for each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_remove : bool
                N booleans to determine if random remove is performed on each sample
            densities : float
                N floats to determine how much to remove from each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_remove[b]:

                    nonzero_indices = self.random_nonzero(image, density=densities[b])
                    image[nonzero_indices] = 0.0

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    def random_nonzero(self, T, density=0.10):
        '''
        Randomly selects nonzero elements

        Arg(s):
            T : torch.Tensor[float32]
                N x C x H x W tensor
            density : float
                percentage of nonzero elements to select
        Returns:
            list[tuple[torch.Tensor[float32]]] : list of tuples of indices
        '''

        # Find all nonzero indices
        nonzero_indices = (T > 0).nonzero(as_tuple=True)

        # Randomly choose a subset of the indices
        random_subset = torch.randperm(nonzero_indices[0].shape[0], device=T.device)
        random_subset = random_subset[0:int(density * random_subset.shape[0])]

        random_nonzero_indices = [
            indices[random_subset] for indices in nonzero_indices
        ]

        return random_nonzero_indices

    def add_noise(self, images_arr, do_add_noise, noise_type, noise_spread):
        '''
        Add noise to images

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_add_noise : bool
                N booleans to determine if noise will be added
            noise_type : str
                gaussian, uniform
            noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
        '''

        for i, images in enumerate(images_arr):
            device = images.device

            for b, image in enumerate(images):
                if do_add_noise[b]:

                    shape = image.shape
                    validity_map = torch.where(
                        image > 0,
                        torch.ones_like(image),
                        torch.zeros_like(image))

                    if noise_type == 'gaussian':
                        image = image + noise_spread * torch.randn(*shape, device=device)
                    elif noise_type == 'uniform':
                        image = image + noise_spread * (torch.rand(*shape, device=device) - 0.5)
                    else:
                        raise ValueError('Unsupported noise type: {}'.format(noise_type))

                    images[b, ...] = image * validity_map

            images_arr[i] = images

        return images_arr


def get_seg_overlay(image_np, seg_np, overlay_opacity=0.5):
    """
    - image_np: HxWx3 uint8 background
    - seg_np: HxW int labels 0,1,2 (classes) or 3 (transparent background)
    - overlay_opacity: opacity of the overlay (0-1.0)
    Classes 0-2 are overlaid at 70% opacity; class 3 is left unchanged.
    """
    palette = {
        1: np.array([  255, 0,   0], dtype=np.uint8),
        2: np.array([  0,   255, 0], dtype=np.uint8),
        3: np.array([0,   0,   255], dtype=np.uint8),
    }
    overlay = image_np.copy().astype(np.float32)
    alpha = overlay_opacity
    for lbl, color in palette.items():
        mask = (seg_np == lbl)
        overlay[mask] = (alpha * color + (1 - alpha) * image_np[mask]).astype(np.uint8)
    return overlay.astype(np.uint8)

# 1) CONFIG
RESTORE_PATH = "trained_models/unet/unet_traintest/model-2200.pth"
OUTPUT_DIR   = "evaluation_results/unet_brightness"
SCAN_FILE    = "training/ergonomics/train_scans_augmented.txt"
MASK_FILE    = "training/ergonomics/train_ground_truths_augmented.txt"

INPUT_SIZE = (640, 480)      # w, h
VAR_INDICES = list(range(11, 22))

# brightness/noise variants
def add_noise(pil_img, sigma):
    arr = np.array(pil_img).astype(np.int16)
    noise = np.random.randn(*arr.shape) * sigma
    arr_n = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_n)

VARIANTS = [
    ("orig",          lambda im: im),
    ("bright+40%",    lambda im: ImageEnhance.Brightness(im).enhance(1.4)),
    ("dark-20%+noise",    lambda im: add_noise(ImageEnhance.Brightness(im).enhance(0.8), 10)),
    ("dark-40%+more",      lambda im: add_noise(ImageEnhance.Brightness(im).enhance(0.6), 20)),
    ("dark-60%+even_more", lambda im: add_noise(ImageEnhance.Brightness(im).enhance(0.4), 30)),
]

# 2) PREP OUTPUT
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(1)

# 3) LOAD VALIDATION LISTS
with open(SCAN_FILE) as f:
    scans = [l.strip() for l in f if l.strip()]
with open(MASK_FILE) as f:
    masks = [l.strip() for l in f if l.strip()]

val_scans = scans[:90]

# 4) LOAD MODEL
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
model = GenericUNetRGBModel(
    encoder_type_segmentation=ENCODER_TYPE_SEGMENTATION,
    n_filters_encoder_segmentation=N_FILTERS_ENCODER_SEGMENTATION,
    decoder_type_segmentation=DECODER_TYPE_SEGMENTATION,
    n_filters_decoder_segmentation=N_FILTERS_DECODER_SEGMENTATION,
    weight_initializer=WEIGHT_INITIALIZER,
    activation_func=ACTIVATION_FUNC,
    use_batch_norm=USE_BATCH_NORM,
    device=device,
)
model.restore_model(RESTORE_PATH)
model.to(device)
model.eval()


# 5) SETUP TRANSFORMS
# The UNet transforms pipeline (only normalization here)
trans = Transforms()


# 6) RUN EVAL LOOP
for idx in VAR_INDICES:
    print(f"> Sample {idx}")
    rgb_p = val_scans[idx]
    base = Image.open(rgb_p).convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)

    overlays = []
    for name, fn in VARIANTS:
        im = fn(base)                 # PIL.Image
        x = normalize(to_tensor(im))  # → Tensor[C,H,W], normalized
        x = x.unsqueeze(0).to(device) # → [1,C,H,W]
        with torch.no_grad():
            out_list = model.forward(x)
            # if the model returned [logits, ...] or (logits, ...), pull out the first entry:
            logits = out_list[0] if isinstance(out_list, (list, tuple)) else out_list

            # now logits is a Tensor and you can interpolate
            logits = F.interpolate(
                logits,
                size=INPUT_SIZE[::-1],
                mode="bilinear",
                align_corners=False
            )
            pred = logits.argmax(dim=1)[0].cpu().numpy()


        overlays.append((name, get_seg_overlay(np.array(im), pred, overlay_opacity=0.5)))

    # 7) PLOT + SAVE
    fig, axs = plt.subplots(1, len(overlays), figsize=(20,4))
    for ax, (name, ov) in zip(axs, overlays):
        ax.imshow(ov)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"unet_{idx}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  saved {out_path}")
