#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:54:08 2023

This script will generate the .npz outputs for:
    - pupil_detection
    - calibration_marker
    - calibration_filter
    - validation_marker
    - validation_filter
    - calibration
    - gaze
    - error
This was originally in "pipelines.py" in vedb_gaze and pipelines.py in 
vedb_extract_gaze.
Notable changes:
    - This will not use a database entry
    - Trying to miminize # of referenced non-public repos
    - Not being set up for Jupyter development
    - Use a YAML/config set up for submitting requests
    - Runs for whole pipeline and provides verbose information about failures

@author: mgreene2
"""

###################
##### Imports #####
###################
import numpy as np
import tqdm
import os
import yaml
import argparse
import vedb_gaze
import utils


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", 
    "--folder", 
    help="input the folder location where all your files are located.")
parser.add_argument(
    "-w",
    "--wholeSession",
    default=True,
    help="Do you want to perform the analysis on the whole session: input should be True or False",
)
parser.add_argument("-l",
                    "--lens", 
                    default=2,
                    help="input which lens was used (1 or 2)")
args = parser.parse_args()

# parse arguments
wholeSession = args.wholeSession
if str(wholeSession).lower() == 'false' or str(wholeSession).lower() == 'f':
    wholeSession = False
elif str(wholeSession).lower() == 'true' or str(wholeSession).lower() == 't':
    wholeSession = True

# Establish default input and output directories
base_dir = '/media/space/Database/staging'  # '/Volumes/space/Database/staging
output_dir = os.path.join(base_dir, args.folder, 'processedGaze')
# make output folder if it doesn't yet exist
os.makedirs(output_dir, exist_ok=True)

# Read in session files
session_folder = args.folder
try:
    world_vid_file = os.path.join(session_folder, 'world.mp4')
    world_time_file = os.path.join(session_folder, 'world_timestamps.npy')
    eye_left_file = os.path.join(session_folder, 'eye1.mp4')
    eye_left_time_file = os.path.join(session_folder, 'eye1_timestamps.npy')
    eye_right_file = os.path.join(session_folder, 'eye0.mp4')
    eye_right_time_file = os.path.join(session_folder, 'eye0_timestamps.npy')
except:
    raise ValueError('Session is missing files!')

# Load necessary files
world_time = np.load(world_time_file)
world_vid_sample = utils.load_mp4(world_vid_file, frames=(
    400, 410))  # chosen arbitrarily to get dims
world_vid_size = world_vid_sample.shape[1:3]
eye_left_time = np.load(eye_left_time_file)
eye_right_time = np.load(eye_right_time_file)

# Get the marker times from previous postprocess.py output
if os.path.exists(os.path.join(args.folder, 'marker_times.yaml')):
    with open(os.path.join(args.folder, 'marker_times.yaml'), 'r') as file:
        marker_times = yaml.safe_load(file)
else:
    # Raise error
    raise ValueError('Please run postprocess.py before running this script.')

# Parse the start and end times
calibration_start = marker_times["calibration_orig_times"][0][0]
calibration_end = marker_times["calibration_orig_times"][0][1]
calibration_start_frame = marker_times["calibration_frames"][0][0]
calibration_end_frame = marker_times["calibration_frames"][0][1]
validation_start_frame = marker_times["validation_frames"][0][0]
validation_end_frame = marker_times["validation_frames"][0][1]
validation_start = marker_times["validation_orig_times"][0][0]
validation_end = marker_times["validation_orig_times"][0][1]
calibration_start_frame_pupil_left, calibration_end_frame_pupil_left = \
    utils.get_frame_indices(calibration_start, calibration_end, eye_left_time)
calibration_start_frame_pupil_right, calibration_end_frame_pupil_right = \
    utils.get_frame_indices(calibration_start, calibration_end, eye_right_time)
validation_start_frame_pupil_left, validation_end_frame_pupil_left = \
    utils.get_frame_indices(validation_start, validation_end, eye_left_time)
validation_start_frame_pupil_right, validation_end_frame_pupil_right = \
    utils.get_frame_indices(validation_start, validation_end, eye_right_time)


#################
# Step 1: find calibration markers
#################

if not os.path.exists(os.path.join(output_dir, 'calibration_markers.npz')):

    print("\n=== Finding calibration markers ===\n")

    calibration_markers = vedb_gaze.marker_detection.find_concentric_circles(world_vid_file, world_time_file, 
                                                        start_frame=calibration_start_frame,
                                                        end_frame=calibration_end_frame,
                                                        progress_bar=tqdm.tqdm)
    # note: might want to change scale to 1 to run on full-size videos.
    # save the calibration markers
    np.savez(os.path.join(output_dir, 'calibration_markers.npz'),
             **calibration_markers)
else:
    calibration_markers = dict(
        np.load(os.path.join(output_dir, 'calibration_markers.npz')))

################
# Step 2: detect pupils during calibration
################

pupil_left_file = os.path.join(output_dir, 'pupil_left_calibrate.npz')
pupil_right_file = os.path.join(output_dir, 'pupil_right_calibrate.npz')

if not os.path.exists(pupil_left_file):

    print("\n=== Finding pupil locations ===\n")

    # get pupil locations during calibration
    pupil_calibration = dict(
        left = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_left_file,
                                                            eye_left_time_file,
                                                            start_frame=calibration_start_frame_pupil_left, 
                                                            end_frame=calibration_end_frame_pupil_left,
                                                            progress_bar=tqdm.tqdm),
        right = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_right_file, 
                                                                eye_right_time_file,
                                                                start_frame=calibration_start_frame_pupil_right, 
                                                                end_frame=calibration_end_frame_pupil_right,
                                                                progress_bar=tqdm.tqdm)
        )
    # get pupil locations during validation
    pupil_validation = dict(
        left = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_left_file,
                                                            eye_left_time_file,
                                                            start_frame=validation_start_frame_pupil_left, 
                                                            end_frame=validation_end_frame_pupil_left,
                                                            progress_bar=tqdm.tqdm),
        right = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_right_file, 
                                                                eye_right_time_file,
                                                                start_frame=validation_start_frame_pupil_right, 
                                                                end_frame=validation_end_frame_pupil_right,
                                                                progress_bar=tqdm.tqdm)
        )

    # combine calibration and validation
    pupil_calibrate = dict(
        left = vedb_gaze.utils.stack_arraydicts(pupil_calibration['left'], 
                                                pupil_validation['left']),
        right = vedb_gaze.utils.stack_arraydicts(pupil_calibration['right'], 
                                                pupil_validation['right']),
    )

    # save results
    np.savez(pupil_left_file, **pupil_calibrate['left'])
    np.savez(pupil_right_file, **pupil_calibrate['right'])

    # get all pupils if running on full session
    if wholeSession:
        pupil = dict(
            left=vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_left_file,
                                                                    eye_left_time_file,
                                                                    progress_bar=tqdm.tqdm),
            right=vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_right_file,
                                                                    eye_right_time_file,
                                                                    progress_bar=tqdm.tqdm)
        )
        # save the output
        np.savez(os.path.join(output_dir, 'pupil_all.npz'), **pupil)

else:
    pupil_calibrate = dict(left=dict(np.load(pupil_left_file, allow_pickle=True)),
                           right=dict(
                               np.load(pupil_right_file, allow_pickle=True)),
                           )

    if wholeSession:
        pupil = dict(np.load(os.path.join(
            output_dir, 'pupil_all.npz'), allow_pickle=True))


################
# Step 3: perform calibration
################

# Note: this next two step is quick, so we're not saving the output

print("\n=== Starting calibration ===\n")

# filter calibration markers for spurious detections
calibration_markers_filtered = vedb_gaze.marker_parsing.find_epochs(
    calibration_markers, world_time)
calibration_input = calibration_markers_filtered[0]  # back to dict

calibration = {}
calibration['left'] = vedb_gaze.calibration.Calibration(pupil_calibrate['left'], 
                                                        calibration_input, 
                                                        (world_vid_size[1], world_vid_size[0]),
                                                        lambd_list=[1e-06,
                                                                    2.9286445646252375e-06,
                                                                    8.576958985908945e-06,
                                                                    2.5118864315095822e-05,
                                                                    7.356422544596421e-05,
                                                                    0.00021544346900318845,
                                                                    0.000630957344480193,
                                                                    0.0018478497974222907,
                                                                    0.0054116952654646375,
                                                                    0.01584893192461114,
                                                                    0.04641588833612782,
                                                                    0.1359356390878527,
                                                                    0.3981071705534969,
                                                                    1.165914401179831,
                                                                    3.414548873833601,
                                                                    10.0],
                                                        max_stds_for_outliers=3.0,
                                                        )
calibration['right'] = vedb_gaze.calibration.Calibration(pupil_calibrate['right'], 
                                                        calibration_input,
                                                        (world_vid_size[1], world_vid_size[0]),
                                                        lambd_list=[1e-06,
                                                                    2.9286445646252375e-06,
                                                                    8.576958985908945e-06,
                                                                    2.5118864315095822e-05,
                                                                    7.356422544596421e-05,
                                                                    0.00021544346900318845,
                                                                    0.000630957344480193,
                                                                    0.0018478497974222907,
                                                                    0.0054116952654646375,
                                                                    0.01584893192461114,
                                                                    0.04641588833612782,
                                                                    0.1359356390878527,
                                                                    0.3981071705534969,
                                                                    1.165914401179831,
                                                                    3.414548873833601,
                                                                    10.0],
                                                        max_stds_for_outliers=3.0,
                                                        )


################
# Step 4: map gaze
################

print("\n=== Gaze mapping ===\n")

gaze_calibration = {}
for lr in ['left', 'right']:
    gaze_calibration[lr] = calibration[lr].map(pupil_calibrate[lr])

# save the output
np.savez(os.path.join(output_dir, 'gaze_calibration.npz'), **gaze_calibration)

# compute all of gaze mapping if whole session
if wholeSession:
    gaze = {}
    for lr in ['left', 'right']:
        gaze[lr] = calibration[lr].map(pupil[lr])

    # save the output
    np.savez(os.path.join(output_dir, 'gaze.npz'), **gaze)


################
# Step 5: find validation markers
################

if not os.path.exists(os.path.join(output_dir, 'validation_markers.npz')):

    print("\n=== Finding validation markers ===\n")
    validation_markers = vedb_gaze.marker_detection.find_checkerboard(world_vid_file, world_time_file,
                                                                      start_frame=validation_start_frame,
                                                                      end_frame=validation_end_frame,
                                                                      progress_bar=tqdm.tqdm)
    # save the calibration markers
    np.savez(os.path.join(output_dir, 'validation_markers.npz'),
             **validation_markers)

else:
    validation_markers = dict(np.load(os.path.join(
        output_dir, 'validation_markers.npz'), allow_pickle=True))


################
# Step 6: compute error
################

if not os.path.exists(os.path.join(output_dir, 'error_validation.npz')):

    print("\n=== Computing error===\n")
    
    # parse which lens was used and use for input into error computation
    lens = args.lens
    if lens==2:
        degrees_horiz = 101
        degrees_vert = 76
    else:
        degrees_horiz = 125
        degrees_vert = 111

    # filter validation markers for spurious detections
    validation_markers_filtered = vedb_gaze.marker_parsing.find_epochs(
        validation_markers, world_time,
        aspect_ratio_threshold=None,  # Don't do aspect ratio thresholding for validation
    )
    # back to dict
    #validation_markers_filtered = validation_markers_filtered[0]

    error_validation = {}
    for lr in ['left', 'right']:
        error_validation[lr] = [vedb_gaze.error_computation.compute_error(ei, gaze_calibration[lr],
                                                                         image_resolution=world_vid_size[::-1],
                                                                         degrees_horiz=degrees_horiz,
                                                                         degrees_vert=degrees_vert,
                                                                         )
                                for ei in validation_markers_filtered]

    if wholeSession:
        error = {}
        for lr in ['left', 'right']:
            error[lr] = [vedb_gaze.error_computation.compute_error(ei, gaze[lr],
                                                                   image_resolution=world_vid_size[::-1],
                                                                   degrees_horiz=degrees_horiz,
                                                                   degrees_vert=degrees_vert,)
                         for ei in validation_markers_filtered]

    # Save the gaze error
    np.savez(os.path.join(output_dir, 'error_validation.npz'), **error_validation)

    if wholeSession:
        np.savez(os.path.join(output_dir, 'error.npz'), **error)
