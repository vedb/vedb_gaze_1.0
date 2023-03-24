# -*- coding: utf-8 -*-

"""
Post-Processing of the eye camera images and T265 kinematic head

# Create/Load the marker_times.yaml file
# Create/Load the odo_times.yaml file

data
@author: 2023-03-18 - Michael Davis
"""

###################
##### Imports #####
###################

# Public
# import argparse
from datetime import datetime
import os
import yaml
from scipy import signal
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import pupil_recording_interface


##########################
##### File Arguments #####
##########################

# Read folder from command-line input
# parser = argparse.ArgumentParser()
# args = parser.parse_args()

# Initialize some things
# TODO: make these file arguments
BASE_DIR = "/Users/mdavis/dev/bates"
INPUT_DIR_NAME = "2022_09_15_15_25_58"
INPUT_DIR = os.path.join(BASE_DIR, INPUT_DIR_NAME)

# Inputs
marker_times_yaml = os.path.join(INPUT_DIR, "marker_times.yaml")
odo_times_yaml = os.path.join(INPUT_DIR, "odo_times.yaml")
world_timestamps = np.load(os.path.join(INPUT_DIR, "world_timestamps.npy"))
odometry_data = pupil_recording_interface.load_dataset(
    INPUT_DIR, odometry="recording", cache=False
)
#################################
##### SPECIFY MARKER EPOCHS #####
#################################

# This is ripped from vedb_store.utils so that we don't have to rely on that library
# https://github.com/vedb/vedb-store/blob/main/vedb_store/utils.py


def specify_marker_epochs(timestamps, yaml_file, fps=30):
    """Manually add the markings for the markers.yaml file
    TODO: add more info here
    """
    # Initialize
    ordinals = ["first", "second", "third", "fourth", "fifth", "too many"]
    marker_type = ["calibration", "validation"]
    markers = {}

    # Take user inputs
    for mk in marker_type:
        markers[mk + "_orig_times"] = []
        for count in ordinals:
            print(f"\n=== {count.capitalize()} {mk} epoch ===")
            minsec_str = input("Please enter start of epoch as `min,sec` : ")
            min_start, sec_start = [float(x) for x in minsec_str.split(",")]
            minsec_str = input("Please enter end of epoch as `min,sec` : ")
            min_end, sec_end = [float(x) for x in minsec_str.split(",")]
            markers[f"{mk}_orig_times"].append(
                [min_start * 60 + sec_start, min_end * 60 + sec_end]
            )
            quit = input(f"Enter additional {mk}? (y/n): ")
            if quit[0].lower() == "n":
                break
        mka = np.array(markers[f"{mk}_orig_times"]).astype(int)
        markers[f"{mk}_frames"] = [[int(a), int(b)] for (a, b) in mka * fps]
        markers[f"{mk}_times"] = [
            [int(np.floor(a)), int(np.ceil(b))] for (a, b) in timestamps[mka * fps]
        ]

    # Save results
    with open(yaml_file, mode="w", encoding="utf-8") as fid:
        yaml.dump(markers, fid)

    return markers


#########################
##### Odometry Plot #####
#########################


def start_end_plot(odometry, marker_times, yaml_file):
    """Use a plot fo find the marker times?
    TODO: add more info here
    """
    # Read in the marker validation times
    all_times = marker_times

    # Convert times
    times = np.array(odometry.time.values)
    time_convert = np.zeros(len(times))
    for i in range(len(times)):
        time_convert[i] = (
            odometry.time.values[i] - odometry.time.values[0]
        ) / 1000000000

    # Plotting accesories
    input_value = all_times["validation_orig_times"][0][1]
    indx = (np.abs(time_convert - input_value)).argmin()
    samps = 200 * 120

    # smooth data for better viewing purposes
    pitch_vel = signal.savgol_filter(odometry.angular_velocity[:, 0], 101, 2)
    yaw_vel = signal.savgol_filter(odometry.angular_velocity[:, 1], 101, 2)

    # Start plotting
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=odometry.time.values,
            y=pitch_vel,
            name="pitch",  # this sets its legend entry
        )
    )
    fig.add_trace(
        go.Scatter(
            x=odometry.time.values, y=yaw_vel, name="yaw",  # this sets its legend entry
        )
    )
    fig.update_layout(
        title="Angular velocity odometry",
        xaxis_title="Time stamps (datetime)",
        yaxis_title="Angular Velocity (radians/second)",
        # xaxis_range=[
        #             odometry.time.values[indx], #this may be added back in the future - sometimes throws an out-of-bounds error
        #             odometry.time.values[indx+samps]] #this may be added back in the future - sometimes throws an out-of-bounds error
    )
    fig.show()

    # Loop
    done = False
    times = []

    while not done:
        pitch_start = input("Pitch Timestamp Start (HH:mm:ss format): ")
        pitch_end = input("Pitch Timestamp End (HH:mm:ss format): ")
        yaw_start = input("Yaw Timestamp Start (HH:mm:ss format): ")
        yaw_end = input("Yaw Timestamp End (HH:mm:ss format): ")
        df_time = pd.Series(odometry.time[0].values)
        pitch_start = datetime.combine(
            df_time.dt.date.values[0],
            datetime.strptime(pitch_start, "%H:%M:%S").time(),
        )
        pitch_end = datetime.combine(
            df_time.dt.date.values[0], datetime.strptime(pitch_end, "%H:%M:%S").time(),
        )
        yaw_start = datetime.combine(
            df_time.dt.date.values[0], datetime.strptime(yaw_start, "%H:%M:%S").time(),
        )
        yaw_end = datetime.combine(
            df_time.dt.date.values[0], datetime.strptime(yaw_end, "%H:%M:%S").time()
        )
        tmp = {
            "calibration": {
                "pitch_start": pitch_start,
                "pitch_end": pitch_end,
                "yaw_start": yaw_start,
                "yaw_end": yaw_end,
            }
        }
        times.append(tmp)
        next_calibration = input("Continue for another pitch/yaw calibration? (y/n) ")
        done = next_calibration != "y"

    # Save output
    with open(yaml_file, mode="w", encoding="utf-8") as fid:
        yaml.dump(times, fid)

    return times


####################################################
##### Load the files for this tracking session #####
####################################################

if __name__ == "__main__":
    # Check folders for existance
    print(f"** Input folder selected: {INPUT_DIR}")
    if not os.path.isdir(INPUT_DIR):
        raise ValueError("Input Folder does not exist. Check the path")

    # Check if markers exist and fill in with world timestamps if not
    if os.path.exists(marker_times_yaml):
        with open(marker_times_yaml, "r", encoding="utf-8") as file:
            marker_times = yaml.safe_load(file)
    else:
        marker_times = specify_marker_epochs(
            timestamps=world_timestamps, yaml_file=marker_times_yaml
        )
    print("** Marker Times Loaded")

    # Execute head calibration and timestamp functions
    if os.path.exists(odo_times_yaml):
        print("** Start & End points for head shakes and nods already exist")
    else:
        print("** Execute head calibration for nods and shakes...")
        times = start_end_plot(
            odometry=odometry_data, marker_times=marker_times, yaml_file=odo_times_yaml
        )
    print("** ODO Times Loaded")

    print("\n** DONE WITH SETTING MARKER TIMES ** \n")
