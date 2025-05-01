import os
import glob
import argparse
from collections import defaultdict


def findSecondToLast(text, pattern):
    '''Returns index of second to last occurrence of pattern in a string'''
    return text.rfind(pattern, 0, text.rfind(pattern))


def isTimestamp(image_filename):
    '''Returns 0 if image_filename is a timestamp, -1 if not'''
    # image will be if timestamp if dash exists
    dash_index = image_filename.rfind('-')
    if dash_index == -1:
        return -1
    return 0


def viewIsTimestamped(image_txt_file):
    '''Returns '''
    # Check first line of txt file: a timestamped first frame should mean all images from
    # that trial are the correct format

    with open(image_txt_file) as f:
        first_line = f.readline()

    return isTimestamp(first_line)


def getTrialName(image_txt_file):
    '''Returns trial name, i.e, 'BP1C' from image textfile path'''
    second_to_last = findSecondToLast(image_txt_file, os.path.sep)

    return image_txt_file[second_to_last + 1:second_to_last + 5]


def main():
    # Given top level directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True, help='path to folder of unzipped data-3d')
    args = parser.parse_args()
    
    # Store images that are timestamped as dict of lists
    # key: trial name, i.e. "BP1C"
    # val: list of timestamped camera image paths 
    timestamped_trials = defaultdict(list)
    two_cam_timestamped_trials = defaultdict(list)

    # Iterate through all image.txt files
    for file in glob.glob(f'{args.data_folder}{os.path.sep}*{os.path.sep}*{os.path.sep}image.txt'):
        if viewIsTimestamped(file) != -1:
            file_trial_name = getTrialName(file)
            timestamped_trials[file_trial_name].append(file)

    # Find all trials where there are > 2 views timestamped
    for trial, views in timestamped_trials.items():
        if len(views) >= 2:
            two_cam_timestamped_trials[trial] = views

    # Human-readable text file output
    with open('./tools/timestamped_trials.txt', 'w') as out:
        for trial, views in two_cam_timestamped_trials.items():
            out.write(trial)
            out.write('\n')
            out.write(f'{str(len(views))} timestamped views')
            out.write('\n')
            out.write(str(views))
            out.write('\n\n')

    out.close()


main()

