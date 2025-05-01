'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import os, sys, glob, argparse, cv2, shutil
import multiprocessing as mp
import numpy as np
sys.path.insert(0, 'src')
import data_utils


'''
Paths for 3d-ergonomics dataset
'''
SYNCHRONIZED_DATA_FILEPATH = os.path.join('setup', 'posenet_frames.txt') # point towards file of stitched images, upload file onto Github
RAW_DATA_DIRPATH = os.path.join('data', 'data-3d') # data directory


'''
Output paths
'''
DERIVED_DATA_DIRPATH = os.path.join(
    'data', 'synchronized_derived')

TRAIN_OUTPUT_REF_DIRPATH = os.path.join('training', 'synchronized')

TRAIN_IMAGE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'train_image.txt')
TRAIN_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'train_depth.txt')
TRAIN_DENSE_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'train_dense_depth.txt')
TRAIN_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'train_intrinsics.txt')

parser = argparse.ArgumentParser()

parser.add_argument('--paths_only', action='store_true')
parser.add_argument('--n_thread',  type=int, default=8)
args = parser.parse_args()


def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple[str]
            image path at time t=0,
            image path at time t=-1 (left),
            image path at time t=1 (right),
            depth path at time t=0,
            depth path at time t=-1 (left),
            depth path at time t=1 (right),
            dense depth path at time t=0,
            dense depth path at time t=-1 (left),
            dense depth path at time t=1 (right),
            intrinsics at time t=0,
            intrinsics at time t=-1 (left),
            intrinsics at time t=1 (right),
            boolean flag if set then create paths only
    Returns:
        str : output concatenated image path at time t=0
        str : output sparse depth path at time t=0
        str : output dense depth path at time t=0
        str : output validity map path at time t=0
        str : output ground truth path at time t=0
    '''

    image0_path, \
        image1_path, \
        image2_path, \
        depth0_path, \
        depth1_path, \
        depth2_path, \
        dense_depth0_path, \
        dense_depth1_path, \
        dense_depth2_path, \
        intrinsics0_path, \
        intrinsics1_path, \
        intrinsics2_path, \
        paths_only = inputs

    # Create validity map and image output path
    image_output_path = image0_path \
        .replace(RAW_DATA_DIRPATH, DERIVED_DATA_DIRPATH)

    # Depth post-processing
    depth_output_path = depth0_path \
        .replace(RAW_DATA_DIRPATH, DERIVED_DATA_DIRPATH)
    
    dense_depth_output_path = dense_depth0_path \
        .replace(RAW_DATA_DIRPATH, DERIVED_DATA_DIRPATH)

    intrinsics_output_path = intrinsics0_path \
        .replace(RAW_DATA_DIRPATH, DERIVED_DATA_DIRPATH)

    # Create output directories
    for output_path in [image_output_path, depth_output_path, dense_depth_output_path, intrinsics_output_path]:
        output_dirpath = os.path.dirname(output_path)
        if not os.path.exists(output_dirpath):
            try:
                os.makedirs(output_dirpath)
            except FileExistsError:
                pass

    if not paths_only:
        # Read images and concatenate together
        # print current directory
        print("Current directory: ", os.getcwd())
        image0 = cv2.imread(image0_path)
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        image = np.concatenate([image1, image0, image2], axis=1)

        depth0 = data_utils.load_depth(depth0_path)
        depth1 = data_utils.load_depth(depth1_path)
        depth2 = data_utils.load_depth(depth2_path)
        depth = np.concatenate([depth1, depth0, depth2], axis=1)
        
        dense_depth0 = data_utils.load_depth(dense_depth0_path)
        dense_depth1 = data_utils.load_depth(dense_depth1_path)
        dense_depth2 = data_utils.load_depth(dense_depth2_path)
        dense_depth = np.concatenate([dense_depth1, dense_depth0, dense_depth2], axis=1)

        intrinsics0 = np.load(intrinsics0_path)
        intrinsics1 = np.load(intrinsics1_path)
        intrinsics2 = np.load(intrinsics2_path)
        intrinsics = np.concatenate([intrinsics1, intrinsics0, intrinsics2], axis=1)

        # Write to disk
        data_utils.save_depth(depth, depth_output_path)
        data_utils.save_depth(dense_depth, dense_depth_output_path)
        cv2.imwrite(image_output_path, image)
        # Only save intrinsics once
        if not os.path.exists(intrinsics_output_path):
            print("Saving intrinsics")
            try:
                np.save(intrinsics_output_path, intrinsics)
            except FileExistsError:
                print("File exists error")
                pass
        else:
            print("os path exists")

    return (image_output_path,
            depth_output_path,
            dense_depth_output_path,
            intrinsics_output_path)


for dirpath in [TRAIN_OUTPUT_REF_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Build a mapping between the camera intrinsics to the directories
image_synchronized_paths = data_utils.read_paths(SYNCHRONIZED_DATA_FILEPATH)

left_image_synchronized_paths = []
right_image_synchronized_paths = []
center_image_synchronized_paths = []

for image_paths in image_synchronized_paths:
    center_image_synchronized_path, left_image_synchronized_path, right_image_synchronized_path = image_paths.split(',')
    left_image_synchronized_paths.append(left_image_synchronized_path)
    right_image_synchronized_paths.append(right_image_synchronized_path)
    center_image_synchronized_paths.append(center_image_synchronized_path)

left_depth_synchronized_paths = [
    image_path.replace('image', 'depth') for image_path in left_image_synchronized_paths
]
right_depth_synchronized_paths = [
    image_path.replace('image', 'depth') for image_path in right_image_synchronized_paths
]
center_depth_synchronized_paths = [
    image_path.replace('image', 'depth') for image_path in center_image_synchronized_paths
]


left_dense_depth_synchronized_paths = [
    image_path.replace('image', 'dense_depth') for image_path in left_image_synchronized_paths
]
right_dense_depth_synchronized_paths = [
    image_path.replace('image', 'dense_depth') for image_path in right_image_synchronized_paths
]
center_dense_depth_synchronized_paths = [
    image_path.replace('image', 'dense_depth') for image_path in center_image_synchronized_paths
]

left_intrinsics_synchronized_paths = []
for image_path in left_image_synchronized_paths:
    new_path = os.path.join(os.path.join(*(image_path.split(os.sep)[0:-2])), 'intrinsics.npy')
    # if originally was absolute path, need to add root back since os.path.join ignores the empty string
    if image_path[0] == os.sep:
        new_path = os.path.join(os.sep, new_path)
    left_intrinsics_synchronized_paths.append(new_path)

right_intrinsics_synchronized_paths = []
for image_path in right_image_synchronized_paths:
    new_path = os.path.join(os.path.join(*(image_path.split(os.sep)[0:-2])), 'intrinsics.npy')
    # if originally was absolute path, need to add root back since os.path.join ignores the empty string
    if image_path[0] == os.sep:
        new_path = os.path.join(os.sep, new_path)
    right_intrinsics_synchronized_paths.append(new_path)

center_intrinsics_synchronized_paths = []
for image_path in center_image_synchronized_paths:
    new_path = os.path.join(os.path.join(*(image_path.split(os.sep)[0:-2])), 'intrinsics.npy')
    # if originally was absolute path, need to add root back since os.path.join ignores the empty string
    if image_path[0] == os.sep:
        new_path = os.path.join(os.sep, new_path)
    center_intrinsics_synchronized_paths.append(new_path)

data_paths = [
    center_image_synchronized_paths,
    left_image_synchronized_paths,
    right_image_synchronized_paths,
    center_depth_synchronized_paths,
    left_depth_synchronized_paths,
    right_depth_synchronized_paths,
    center_dense_depth_synchronized_paths,
    left_dense_depth_synchronized_paths,
    right_dense_depth_synchronized_paths,
    center_intrinsics_synchronized_paths,
    left_intrinsics_synchronized_paths,
    right_intrinsics_synchronized_paths
]

train_image_output_paths = []
train_depth_output_paths = []
train_dense_depth_output_paths = []
train_intrinsics_output_paths = []

for paths in zip(*data_paths):
    # Process paths: stitch together 
    image_output_path, depth_output_path, dense_depth_output_path, intrinsics_output_path = process_frame(paths + tuple([args.paths_only]))

    train_image_output_paths.append(image_output_path)
    train_depth_output_paths.append(depth_output_path)
    train_dense_depth_output_paths.append(dense_depth_output_path)
    train_intrinsics_output_paths.append(intrinsics_output_path)


# Write all training file paths
print('Storing training image file paths into: {}'.format(
    TRAIN_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGE_OUTPUT_FILEPATH, train_image_output_paths)

print('Storing training sparse depth file paths into: {}'.format(
    TRAIN_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_DEPTH_OUTPUT_FILEPATH, train_depth_output_paths)

print('Storing training dense depth file paths into: {}'.format(
    TRAIN_DENSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_DENSE_DEPTH_OUTPUT_FILEPATH, train_dense_depth_output_paths)

print('Storing training intrinsics file paths into: {}'.format(
    TRAIN_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_OUTPUT_FILEPATH, train_intrinsics_output_paths)
