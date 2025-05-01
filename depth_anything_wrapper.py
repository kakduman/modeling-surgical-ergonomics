import os
import sys
import cv2
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
depth_anything_dir = os.path.join(current_dir, "Depth-Anything-V2-main")
if depth_anything_dir not in sys.path:
    sys.path.insert(0, depth_anything_dir)

# Import the DepthAnythingV2 model from the Depth-Anything-V2 code.
from depth_anything_v2.dpt import DepthAnythingV2

class DepthAnythingWrapper:
    """
    A lightweight wrapper for the Depth-Anything-V2 model focused on forward image inference.
    """
    def __init__(self, model_weights_path=None, encoder='vits', device=None):
        # Set up device (CUDA, MPS, or CPU)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() 
                                  else 'mps' if torch.backends.mps.is_available() 
                                  else 'cpu')
        self.device = device

        # For 'vits' encoder, use smaller feature sizes
        if encoder == 'vits':
            features = 64  # Smaller feature size for vits
            out_channels = [48, 96, 192, 384]  # Smaller channel sizes
        elif encoder == 'vitb':
            features = 128
            out_channels = [96, 192, 384, 768]
        else: # for vitl
            features = 256  # Default
            out_channels = [256, 512, 1024, 1024]  # Default

        # Instantiate and configure the model.
        self.model = DepthAnythingV2(
            encoder=encoder,
            features=features,
            out_channels=out_channels,)
        self.model.to(self.device)
        self.model.eval()

        # Optionally load pretrained weights.
        if model_weights_path is not None:
            state = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded weights from {model_weights_path}")

    def infer(self, image_path, input_size=518): # wrap back to forward
        """
        Perform depth estimation on a given image.

        Args:
            image_path (str): Path to the input image (PNG, JPEG, etc.).
            input_size (int): Target size for the model input (default: 518).

        Returns:
            np.ndarray: The inferred depth map.
        """
        # Read the image using OpenCV (which loads in BGR format).
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Image not found or unable to load: {image_path}")

        # Use the model's inference function (which handles resizing, etc.).
        depth_map = self.model.infer_image(raw_image, input_size=input_size)
        return depth_map
