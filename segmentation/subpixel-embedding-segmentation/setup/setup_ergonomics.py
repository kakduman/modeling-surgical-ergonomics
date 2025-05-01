# setup the data for the ergonomics segmentation by converting the segmentation masks (in png form) to numpy arrays
# assumes directory structure is already set up (alternatively, could just reuse the same directory)

# assume this script is called in the subpixel directory, i.e. setup/setup_ergonomics.py

import argparse
from PIL import Image
import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True, help='path to file containing paths to segmentation masks')
parser.add_argument('--labels_file', type=str, required=True, help='path to the file containing RGB labels for each class')

args = parser.parse_args()

label_list = []

with open(args.labels_file, 'r') as f:
    for line in f:
        num_list = line.strip().split(',')
        label_list.append(np.array(num_list).astype(np.float32))

original_folder = 'annotated_png'
new_folder = 'annotated'
image_extension = '.png'

with open(args.input_file, 'r') as f:
    for line in f:
        path = line.strip()
        image = Image.open(path).convert('RGB')
        image_array = np.asarray(image, np.float32)

        image_name = path.split(os.sep)[-1]

        # assumption is that the background is zeros
        # output is H * W * 3
        # since I do 3-class segmentation, convert colors to R, G, B
        output_array = np.zeros([image_array.shape[0], image_array.shape[1], 3])

        new_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        for i in range(len(label_list)):
            output_array[np.where((image_array == label_list[i]).all(axis=2))] = new_colors[i]

        path = path.replace(image_extension, '')
        path = path.replace(original_folder, new_folder)

        path_dir = path[:path.rfind(os.sep)]
        if not os.path.exists(path_dir):
            try:
                os.makedirs(path_dir)
            except FileExistsError:
                pass

        np.save(path, output_array)

# create the text file containing the paths to the RGB annotations
ground_truths_file_name = 'training/ergonomics/train_ground_truths.txt'
with open(args.input_file, 'r') as f:
    new_contents = (f.read().replace(original_folder, new_folder)).replace('png', 'npy')
with open(ground_truths_file_name, 'w') as f:
    f.write(new_contents)
