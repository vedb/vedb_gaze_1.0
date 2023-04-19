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
parser.add_argument("-f", "--folder", help = "input the folder location where all your files are located.")
parser.add_argument(
    "-w",
    "--wholeSession",
    default=True,
    help="Do you want to perform the analysis on the whole session: input should be True or False",
)
args = parser.parse_args()

# Establish default input and output directories
base_dir = '/media/space/Database/staging' # '/Volumes/space/Database/staging
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
world_vid_sample = utils.load_mp4(world_vid_file, frames=(400, 410)) #chosen arbitrarily to get dims
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
                                                                scale=0.5, 
                                                                progress_bar=tqdm.tqdm)
    # note: might want to change scale to 1 to run on full-size videos. 
    # save the calibration markers
    np.savez(os.path.join(output_dir, 'calibration_markers.npz'), **calibration_markers)
    
else:
    calibration_markers = np.load(os.path.join(output_dir, 'calibration_markers.npz'), allow_pickle=True)

################
# Step 2: detect pupils during calibration
################

if not os.path.exists(os.path.join(output_dir, 'pupil_calibration.npz')):
    
    print("\n=== Finding pupil locations ===\n")
    
    pupil_calibration = dict(
        left = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_left_file, eye_left_time_file, 
                                                                 start_frame=calibration_start_frame_pupil_left,
                                                                 end_frame=calibration_end_frame_pupil_left,
                                                                 progress_bar=tqdm.tqdm,),
        right = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_right_file, eye_right_time_file, 
                                                                 start_frame=calibration_start_frame_pupil_right,
                                                                 end_frame=calibration_end_frame_pupil_right,
                                                                 progress_bar=tqdm.tqdm,)
        )
    
    pupil_validation = dict(
        left = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_left_file, eye_left_time_file, 
                                                                 start_frame=validation_start_frame_pupil_left,
                                                                 end_frame=validation_end_frame_pupil_left,
                                                                 progress_bar=tqdm.tqdm,),
        right = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_right_file, eye_right_time_file, 
                                                                 start_frame=validation_start_frame_pupil_right,
                                                                 end_frame=validation_end_frame_pupil_right,
                                                                 progress_bar=tqdm.tqdm,)
        )
    
    # Save the pupil calibration files
    np.savez(os.path.join(output_dir, 'pupil_calibration.npz'), **pupil_calibration)
    np.savez(os.path.join(output_dir, 'pupil_validation.npz'), **pupil_validation)
    
    # get all pupils if running on full session
    if args.wholeSession:
        pupil = dict(
            left = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_left_file, eye_left_time_file, 
                                                                     progress_bar=tqdm.tqdm,),
            right = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_right_file, eye_right_time_file, 
                                                                     progress_bar=tqdm.tqdm,)
            )
        
        np.savez(os.path.join(output_dir, 'pupil_all.npz'), **pupil)
        
        # Combine pupils into single dictionary
        pupil_calibration = dict(
                        left = vedb_gaze.utils.stack_arraydicts(pupil_calibration['left'], 
                                                                pupil_validation['left']),
                        right = vedb_gaze.utils.stack_arraydicts(pupil_calibration['right'], 
                                                                pupil_validation['right']),
                    )
        
else:
    pupil_calibration = np.load(os.path.join(output_dir, 'pupil_calibration.npz'), allow_pickle=True)
    pupil_validation = np.load(os.path.join(output_dir, 'pupil_validation.npz'), allow_pickle=True)
    if args.wholeSession:
        pupil = np.load(os.path.join(output_dir, 'pupil_all.npz'), allow_pickle=True)
        

################
# Step 3: perform calibration
################

if not os.path.exists(os.path.join(output_dir, 'calibration.npz')):

    print("\n=== Starting calibration ===\n")
    
    # step 1: filter for spurious detections
    calibration_markers_filtered = vedb_gaze.marker_parsing.find_epochs(
            calibration_markers, world_time)
    calibration_markers_filtered = calibration_markers_filtered[0] # back to dict
    
    calibration = {}
    calibration['left'] = vedb_gaze.calibration.Calibration(pupil_calibration['left'], 
                                                         calibration_markers_filtered, 
                                                         (world_vid_size[1], world_vid_size[0]), 
                                                         calibration_type='monocular_pl',
                                                         max_stds_for_outliers=3.0,)
    calibration['right'] = vedb_gaze.calibration.Calibration(pupil_calibration['right'], 
                                                          calibration_markers_filtered, 
                                                          (world_vid_size[1], world_vid_size[0]),
                                                          calibration_type='monocular_pl',
                                                          max_stds_for_outliers=3.0,)
    
    # Save the calibration files
    np.save(os.path.join(output_dir, 'calibration.npy'), **calibration)

else:
    calibration = np.load(os.path.join(output_dir, 'calibration.npz'), allow_pickle=True)
    


################
# Step 4: map gaze
################

if not os.path.exists(os.path.join(output_dir, 'gaze_calibration.npz')):

    print("\n=== Gaze mapping ===\n")
    
    gaze_calibration = {}
    for lr in ['left', 'right']:
        gaze_calibration[lr] = calibration[lr].map(pupil_calibration[lr])
    
    # compute all of gaze mapping if whole session
    if args.wholeSession:
        gaze = {}
        for lr in ['left','right']:
            gaze[lr] = calibration[lr].map(pupil[lr])

    
    # Save the gaze mappings
    np.savez(os.path.join(output_dir, 'gaze_calibration.npz'), **gaze_calibration)
    if args.wholeSession:
        np.savez(os.path.join(output_dir, 'gaze.npz'), **gaze)
        
else:
    gaze_calibration = np.load(os.path.join(output_dir, 'gaze_calibration.npz'), allow_pickle=True)
    if args.wholeSession:
        gaze = np.load(os.path.join(output_dir, 'gaze.npz'), allow_pickle=True)


################
# Step 5: find validation markers
################

if not os.path.exists(os.path.join(output_dir, 'validation_markers.npz')):

    print("\n=== Finding validation markers ===\n")
    validation_markers = vedb_gaze.marker_detection.find_checkerboard(world_vid_file, world_time_file, 
                                                                start_frame=validation_start_frame, 
                                                                end_frame=validation_end_frame, 
                                                                scale=0.5, 
                                                                progress_bar=tqdm.tqdm)
    # save the calibration markers
    np.savez(os.path.join(output_dir, 'validation_markers.npz'), **validation_markers)
    
else:
    validation_markers = np.load(os.path.join(output_dir, 'validation_markers.npz'), allow_pickle=True)

################
# Step 6: compute error
################

if not os.path.exists(os.path.join(output_dir, 'error_validation.npz')):

    print("\n=== Computing error===\n")
    
    # filter validation markers for spurious detections
    validation_markers_filtered = vedb_gaze.marker_parsing.find_epochs( 
        validation_markers, world_time, 
        aspect_ratio_threshold=None, # Don't do aspect ratio thresholding for validation
        )
    
    error_validation = {}
    for lr in ['left', 'right']:
        error_validation[lr] = [vedb_gaze.error_computation.compute_error(ei, gaze_calibration[lr], 
                                                          image_resolution=world_vid_size[::-1], 
                                                          lambd=1.0, )
                                for ei in validation_markers_filtered]
    
    if args.wholeSession:
        error = {}
        for lr in ['left', 'right']:
            error[lr] = [vedb_gaze.error_computation.compute_error(ei, gaze[lr], 
                                                          image_resolution=world_vid_size[::-1], 
                                                          lambd=1.0, )
                         for ei in validation_markers_filtered]
    
    # Save the gaze error
    np.savez(os.path.join(output_dir, 'error_validation.npz'), **error_validation)
    
    if args.wholeSession:
        np.savez(os.path.join(output_dir, 'error.npz'), **error)
