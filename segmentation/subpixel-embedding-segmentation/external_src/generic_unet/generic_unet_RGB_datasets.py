import numpy as np
import torch.utils.data
import sys
from PIL import Image

sys.path.insert(0, 'src')


class GenericUNetRGBTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for U-Net to fetch
    (1) input images
    (2) ground truth annotations

    Arg(s):
        image_paths : list[str]
            list of paths to images
        ground_truth_paths : list[str]
            list of paths to ground truth annotations
        shape : tuple[int]
            (n_height, n_width) tuple
    '''
    def __init__(self,
                 image_paths,
                 ground_truth_paths,
                 shape):

        # Dataset paths and shape
        self.image_paths = image_paths
        self.n_sample = len(image_paths)

        self.n_height = shape[0]
        self.n_width = shape[1]

        self.ground_truth_paths = ground_truth_paths

        self.color_to_class_map = {
            (0, 0, 0): 0,
            (1, 0, 0): 1,
            (0, 1, 0): 2,
            (0, 0, 1): 3
        }

    def __getitem__(self, index):
        '''
        Fetches scan and ground truth annotation

        Arg(s):
            index : int
                index of sample in dataset
        Returns
            numpy[float32] : 3 x H x W input image
            numpy[uint64] : 1 x H x W ground truth annotations
        '''

        # Each input image is H x W x 3
        image = Image.open(self.image_paths[index]).convert('RGB')
        image = np.asarray(image, np.float32)

        # Each ground-truth annotation is H x W x 3
        ground_truth = np.load(self.ground_truth_paths[index]).astype(np.float32)

        # label map: H x W x 1, with values [0, 4] for 3 classes and 1 background
        label = np.zeros(image.shape[0:2])

        for color, class_label in self.color_to_class_map.items():
            label[np.where((ground_truth == color).all(axis=2))] = class_label

        label = label.reshape((label.shape[0], label.shape[1], 1))

        # put channel dimension first
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        return image.astype(np.float32), label.astype(np.int64)

    def __len__(self):
        return self.n_sample


class GenericUNetRGBInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for U-Net to fetch input images

    Arg(s):
        image_paths : list[str]
            list of paths to images
        shape : tuple[int]
            (n_height, n_width) tuple
    '''
    def __init__(self, image_paths, shape):

        # Dataset paths and shape
        self.image_paths = image_paths
        self.n_sample = len(image_paths)

        self.n_height = shape[0]
        self.n_width = shape[1]

    def __getitem__(self, index):
        '''
        Fetches input image

        Arg(s):
            index : int
                index of sample in dataset
        Returns
            numpy[float32] : 3 x H x W input image
        '''

        # Each input image is H x W x 3
        image = Image.open(self.image_paths[index]).convert('RGB')
        image = np.asarray(image, np.float32)
        image = np.transpose(image, (2, 0, 1))

        return image.astype(np.float32)

    def __len__(self):
        return self.n_sample
