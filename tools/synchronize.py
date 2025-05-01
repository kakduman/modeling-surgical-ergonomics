
import argparse
import os


# any timestamps greater than a certain distance apart should not be associated, regardless of how "close" they are
# timestamps appear to be in milliseconds; camera FPS is 30, so window should be less than ~33 ms
MAX_DIST = 25


# output file name to be input into the next step (setup)
output_file_name = 'tools/posenet_frames.txt'

# absolute path for Google Drive
absolute_file_path_prefix = ''

# set up args
parser = argparse.ArgumentParser()
parser.add_argument('--operation_name', type=str, required=True, help='name of the operation: SB, ON, VS, BP')
parser.add_argument('--data_dir', type=str, required=False, help='directory path to the data location')
# because most folders only have 2/3 folders with timestamps, require only 2 cameras instead of 3
parser.add_argument('--truth_cam_info', type=str, required=True, help='path to info file of "truth" camera view')
parser.add_argument('--other_cam_info', type=str, required=True, help='path to info file of "other" (left) camera view')
# optional arg for the 3rd camera
# if there is a third camera, truth = back, other = left, third = right
parser.add_argument('--third_cam_info', type=str, required=False, help='path to info file of the third (right) camera view')

args = parser.parse_args()

# constant directory path to the data location in Google Drive
if args.data_dir is None:
    DIR_PATH = "data/data-3d/" + args.operation_name
else:
    data_dir = args.data_dir
    if data_dir[-1] == os.sep:
        data_dir = data_dir[:-1]
    DIR_PATH = os.path.join(data_dir, args.operation_name)

# add the Google Drive prefix
DIR_PATH = os.path.join(absolute_file_path_prefix, DIR_PATH)


def getTimestamp(image_name):
    """Converts image filename (e.g., '0000016151-1681418308252.8262.png')
    to timestamp (e.g., '1681418308252.8262'). Returns -1 on failure."""
    dash_index = image_name.rfind('-')
    if dash_index == -1:
        return -1
    # remove file extension
    dot_index = image_name.rfind('.')
    return float(image_name[dash_index + 1:dot_index])


def getClosestImage(true_timestamp, image_list, list_index):
    """Finds closest image by timestamp (from another camera viewpoint) to the true_timestamp, where the closest image
    occurs before the true_timestamp and is within some window distance from the true_timestamp
    Returns the index of the closest image in image_list as well as the list index to start from next function call."""

    closest_index = -1

    if list_index >= len(image_list):
        return closest_index, list_index

    # iterate through the other images until the timestamp is greater; since increasing, always do better
    current_timestamp = getTimestamp(image_list[list_index])
    while list_index < len(image_list) and current_timestamp <= true_timestamp:
        if current_timestamp + MAX_DIST >= true_timestamp:
            closest_index = list_index
        list_index += 1
        if list_index >= len(image_list):
            break
        current_timestamp = getTimestamp(image_list[list_index])

    return closest_index, list_index


def main():
    # we want to read the text file listing all the image files, e.g. image.txt or depth.txt
    image_file_true = args.truth_cam_info
    image_file_other = args.other_cam_info

    # no third camera
    if args.third_cam_info is None:

        with open(image_file_true) as f:
            image_file_list_true = f.readlines()

        with open(image_file_other) as f:
            image_file_list_other = f.readlines()

        # End result: 2D array of filenames matched to the "true" timestamp
        associated_image_map = []

        # indices into the other image lists
        list_index_other = 0

        for image_file in image_file_list_true:
            cur_timestamp = getTimestamp(image_file)

            closest_index_other, list_index_other = getClosestImage(cur_timestamp, image_file_list_other, list_index_other)

            if closest_index_other != -1:
                associated_image_map.append([image_file, image_file_list_other[closest_index_other]])
    else:
        image_file_third = args.third_cam_info

        with open(image_file_true) as f:
            image_file_list_true = f.readlines()

        with open(image_file_other) as f:
            image_file_list_other = f.readlines()

        with open(image_file_third) as f:
            image_file_list_third = f.readlines()

        # End result: 2D array of filenames matched to the "true" timestamp
        associated_image_map = []

        # indices into the other image lists
        list_index_other = 0
        list_index_third = 0

        for image_file in image_file_list_true:
            cur_timestamp = getTimestamp(image_file)

            closest_index_other, list_index_other = getClosestImage(cur_timestamp, image_file_list_other,
                                                                    list_index_other)

            closest_index_third, list_index_third = getClosestImage(cur_timestamp, image_file_list_third,
                                                                    list_index_third)

            # output format in truth, other third form
            if closest_index_other != -1 and closest_index_third != -1:
                associated_image_map.append([image_file, image_file_list_other[closest_index_other], image_file_list_third[closest_index_third]])

    with open(output_file_name, 'w') as f:
        for elem_list in associated_image_map:
            output_list = []
            for item in elem_list:
                # remove whitespace
                item = item.strip()
                # remove existing separators
                if '\\' in item:
                    item_as_list = item.split('\\')
                else:
                    item_as_list = item.split('/')
                # remove existing prefix
                # assume data in the format ...extraneous.../SB1Cleft/data/image/0000000299-1681416360811.0564.png
                item_as_list = item_as_list[-4:]
                # add our own separators, for consistency
                item_as_str = os.path.join(*item_as_list)
                # prefix the data with the DIR PATH
                item_as_str = os.path.join(DIR_PATH, item_as_str)
                output_list.append(item_as_str)
            # separate associated output files with a comma
            output_line = ','.join(output_list)
            f.write(output_line)
            f.write('\n')


main()
