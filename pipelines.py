# -*- coding: utf-8 -*-
"""
This script will generate the .npz outputs for:
    - pupil_detection
    - calibration_marker
    - calibration_filter
    - validation_marker
    - validation_filter
    - calibration
    - gaze
    - error

This was originally in "pipelines.py" in vedb_gaze.
Notable changes:
    - This will not use a database entry
    - Trying to miminize # of referenced non-public repos
    - Not being set up for Jupyter development
    - Use a YAML/config set up for submitting requests

data
@author: Michael Davis 2023-02-22
"""

###################
##### Imports #####
###################

# Freely available
import argparse
import os
import functools
import pprint
import tqdm
import numpy as np
import pydra
import yaml
from vedb_gaze import utils
from vedb_gaze import calibration
from vedb_gaze import file_io

# Initialization
BASE_DIR = "/media/space/Database/"
CODE_DIR = '.'
PYDRA_OUTPUT_DIR = os.path.join(BASE_DIR, "db/processed/gaze/")
pipeline_name = "vedb_pipeline"

##########################
##### File Arguments #####
##########################

# Read folder from command-line input
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help = "input the folder location where all your files are located.")
parser.add_argument(
    "-w",
    "--wholeSession",
    default=False,
    help="Do you want to perform the analysis on the whole session: input should be True or False",
)
args = parser.parse_args()

# error if no folder
if not args.folder:
    raise ValueError('Please indicate session folder YYYY-MM-DD-HH-MM-SS')
else:
    INPUT_DIR = args.folder
    INPUT_DIR_NAME = INPUT_DIR.split('/')[-1]

# Run full session
run_full_session = False
if args.wholeSession:
    if (args.wholeSession is True) or str(args.wholeSession).lower().startswith("t"):
        run_full_session = True


##########################
##### YAML Functions #####
##########################


def read_update_YAML_config(
    pipeline_yaml, final_yaml, marker_yaml, show_output=False, save_output=True
):
    """
    The YAML ""creates"" the Pydra pipeline, and lets us define some extra fn_inputs
    The YAML also specifies where to get the most of the config inputs for each step
    So add the fn_inputs from the pipeline YAML to the config YAMLs
    """
    # Inputs
    with open(pipeline_yaml, "r", encoding="utf-8") as fid:
        pipe_config = yaml.safe_load(fid)

    # loop through each step and add the information from the configs
    for pipe_step, param_dict in pipe_config.items():
        # Start with the config options
        if param_dict["fn_config"] is not None:
            step_config_fname = (
                f'{param_dict["fn_name"]}-{param_dict["fn_config"]}.YAML'
            )
            step_config_yaml = os.path.join(PARAM_DIR, step_config_fname)
            with open(step_config_yaml, "r", encoding="utf-8") as fid:
                step_config = yaml.safe_load(fid)
        else:
            step_config = dict()

        # Have to check type first because some inputs are blanks
        if isinstance(param_dict["fn_inputs"], dict):
            # Replace progress bar boolean with actual progress bar code
            if "progress_bar" in param_dict["fn_inputs"].keys():
                if param_dict["fn_inputs"]["progress_bar"]:
                    param_dict["fn_inputs"]["progress_bar"] = tqdm.tqdm

            # for calib marker detection and valid marker detection
            if os.path.exists(marker_yaml):
                # read in the yaml
                with open(marker_yaml, "r", encoding="utf-8") as fid:
                    marker_times = yaml.safe_load(fid)

                # calibration
                if pipe_step == "calibration_marker_detection":
                    param_dict["fn_inputs"]["start_frame"] = marker_times[
                        "calibration_frames"
                    ][0][0]
                    param_dict["fn_inputs"]["end_frame"] = marker_times[
                        "calibration_frames"
                    ][0][1]
                # validation
                elif pipe_step == "validation_marker_detection":
                    param_dict["fn_inputs"]["start_frame"] = marker_times[
                        "validation_frames"
                    ][0][0]
                    param_dict["fn_inputs"]["end_frame"] = marker_times[
                        "validation_frames"
                    ][0][1]

            # Add the fn_inputs from pipeline config to the step config
            # TODO: handle collisions?
            param_dict["fn_inputs"].update(step_config)
        else:
            param_dict["fn_inputs"] = step_config

        # Add the output for Pydra
        # TODO: handle cases when script partially completes
        param_dict["output_file"] = os.path.join(
            PYDRA_OUTPUT_SESSION_DIR,
            f"{pipe_step}-{param_dict['fn_name']}-{param_dict['fn_config']}.npz",
        )

    if show_output:
        pprint.pprint(pipe_config, sort_dicts=False)

    if save_output:
        with open(final_yaml, mode="w", encoding="utf-8") as fid:
            yaml.dump(pipe_config, fid, sort_keys=False)
            print("** Saved session variables")

    return pipe_config


#######################
##### Pydra Steps #####
#######################


def add_basic_steps(func):
    """
    The point here is to do the same process for each step in the pipeline
    the "func" part inserts the code specific to each of the pipeline sections
    For more info on how decorators work, go here
    https://realpython.com/primer-on-python-decorators/
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fpath = pipeline_config[pydra_step]["output_file"]

        # Skip if it already exists
        if os.path.exists(fpath):
            print(f"** {pydra_step.upper()} previously completed")

        else:
            print(f"** Starting {pydra_step.upper()}")
            # Specific to this one
            kwargs = func(*args, **kwargs)

            # Update with whatever was in the pipeline config
            pipeline_kwargs = pipeline_config[pydra_step]["fn_inputs"]
            kwargs.update(pipeline_kwargs)
            print(f"{pydra_step} kwargs: {kwargs.keys()}")

            # Run the function specified in the pipeline
            pipeline_func = f'{pipeline_config[pydra_step]["fn_module"]}.{pipeline_config[pydra_step]["fn_file"]}.{pipeline_config[pydra_step]["fn_name"]}'
            pipeline_func = utils.get_function(pipeline_func)
            data = pipeline_func(**kwargs)

            # Special consideration for steps
            if pydra_step in ["pupil_detection_left", "pupil_detection_right"]:
                # Special consideration for pupil detection QC analysis
                if run_full_session is False:
                    kwargs["start_frame"] = kwargs["valid_st"]
                    kwargs["end_frame"] = kwargs["valid_end"]
                    data2 = pipeline_func(**kwargs)
                    data = utils.stack_arraydicts(data, data2)
            if pydra_step == "calibration_marker_filtering":
                data = data[calibration_epoch]
            elif pydra_step == "validation_marker_filtering":
                # TODO: handle len(0) error
                data = data[validation_epoch]
            elif pydra_step in ["calibrate_left", "calibrate_right"]:
                data = data.calibration_data

            # print(data)
            np.savez(fpath, **data)
            print(f"** {pydra_step.upper()} now complete. Saved to {fpath}")

        return fpath

    return wrapper


@pydra.mark.task
@add_basic_steps
def pupil_detection(eye):
    """
    Code specific to the pupil detection portion
    Reused for both eyes

    Parameters
    ----------
    eye : str (left/right)
    """

    kwargs = dict()

    if eye == "left":
        kwargs["video_file"] = eye_left_video_file
        kwargs["timestamp_file"] = eye_left_time_file
    elif eye == "right":
        kwargs["video_file"] = eye_right_video_file
        kwargs["timestamp_file"] = eye_right_time_file

    if run_full_session is True:
        kwargs["start_frame"] = None
        kwargs["end_frame"] = None
    else:
        eye_timestamps = np.load(kwargs["timestamp_file"])
        with open(marker_times_yaml, "r", encoding="utf-8") as fid:
            marker_times = yaml.safe_load(fid)
        calib_st, calib_end = utils.get_frame_indices(
            *marker_times["calibration_times"][calibration_epoch], eye_timestamps
        )
        valid_st, valid_end = utils.get_frame_indices(
            *marker_times["validation_times"][validation_epoch], eye_timestamps
        )

        kwargs["start_frame"] = calib_st
        kwargs["end_frame"] = calib_end
        kwargs["valid_st"] = valid_st
        kwargs["valid_end"] = valid_end

    return kwargs


@pydra.mark.task
@add_basic_steps
def marker_detection():
    """
    Code specific to the calibration & validation marker detection

    Parameters
    ----------
    marker_type : str (calibration/validation)
    """
    kwargs = dict()

    kwargs["timestamp_file"] = world_time_file
    kwargs["video_file"] = world_video_file

    return kwargs


@pydra.mark.task
@add_basic_steps
def marker_filtering(marker_fpath):
    """
    Code specific to the calibration marker filtering

    Parameters
    ----------
    markers : Pydra Output from Previous Step
    """

    kwargs = dict()

    kwargs["all_timestamps"] = np.load(world_time_file)
    kwargs["marker"] = np.load(marker_fpath, allow_pickle=True)

    return kwargs


@pydra.mark.task
@add_basic_steps
def calibrate(marker_fpath, pupil_fpath):
    """
    Code specific to the calibrate step
    Reused for both eyes

    Parameters
    ----------
    eye : str (left/right)

    need for calibration class
    pupil arrays --> from pupil detection
    calibration_arrays --> from cal_epoch_choice
    video_dims --> below
    """

    kwargs = dict()

    kwargs["pupil_arrays"] = dict(np.load(pupil_fpath, allow_pickle=True))
    kwargs["calibration_arrays"] = dict(np.load(marker_fpath, allow_pickle=True))
    kwargs["video_dims"] = vdims

    return kwargs


@pydra.mark.task
@add_basic_steps
def map_gaze(calibration_fpath, pupil_fpath):
    """
    Code specific to the gaze step

    Parameters
    ----------
    eye : str (left/right)

    need for calibration class
    pupil arrays --> from pupil detection
    calibration --> from previous step
    """

    kwargs = dict()

    kwargs["pupil_data"] = dict(np.load(pupil_fpath, allow_pickle=True))
    kwargs["calibration"] = calibration.Calibration.load(calibration_fpath)

    return kwargs


@pydra.mark.task
@add_basic_steps
def compute_error(marker_fpath, gaze_fpath):
    """
    Code specific to the compute error step

    Parameters
    ----------
    eye : str (left/right)

    need for compute error class
    gaze --> gaze for that eye
    marker --> validation markers filtered
    """

    kwargs = dict()

    kwargs["marker"] = dict(np.load(marker_fpath, allow_pickle=True))
    kwargs["gaze"] = dict(np.load(gaze_fpath, allow_pickle=True))

    return kwargs


# Initialize some things
PARAM_DIR = os.path.join(INPUT_DIR, "configs")
PYDRA_OUTPUT_SESSION_DIR = os.path.join(PYDRA_OUTPUT_DIR, INPUT_DIR_NAME)

# Input fpaths for pipline metadata
pipeline_config_yaml = os.path.join(
    CODE_DIR, f"{pipeline_name}_script.yaml"
)
session_config_yaml = os.path.join(PYDRA_OUTPUT_SESSION_DIR, "session_config.yaml")


# Input fpaths for pipeline steps
marker_times_yaml = os.path.join(INPUT_DIR, "marker_times.yaml")
config_yaml = os.path.join(INPUT_DIR, "config.yaml")

eye_left_video_file = os.path.join(INPUT_DIR, "eye1.mp4")
eye_right_video_file = os.path.join(INPUT_DIR, "eye0.mp4")

eye_left_time_file = os.path.join(INPUT_DIR, "eye1_timestamps.npy")
eye_right_time_file = os.path.join(INPUT_DIR, "eye0_timestamps.npy")

world_video_file = os.path.join(INPUT_DIR, "world.mp4")
world_time_file = os.path.join(INPUT_DIR, "world_timestamps.npy")

vdims = file_io.var_size(world_video_file)[2:0:-1]

with open(config_yaml, "r", encoding="utf-8") as fid:
    input_config = yaml.safe_load(fid)

world_camera_resolution = utils.parse_resolution(
    input_config["streams"]["video"]["world"]["resolution"]
)

calibration_epoch = 0
validation_epoch = 0


####################
##### RUN TIME #####
####################

if __name__ == "__main__":

    # Check folders for existance
    if not os.path.exists(PYDRA_OUTPUT_DIR):
        os.mkdir(PYDRA_OUTPUT_DIR)

    # Create a session dir for this session
    if not os.path.exists(PYDRA_OUTPUT_SESSION_DIR):
        os.mkdir(PYDRA_OUTPUT_SESSION_DIR)
        print("** Created sub-directory for this session in the Tmp Pydra Output dir")

    # Read in the YAML config file and update it with the param tags listed
    pipeline_config = read_update_YAML_config(
        pipeline_yaml=pipeline_config_yaml,
        final_yaml=session_config_yaml,
        marker_yaml=marker_times_yaml,
        show_output=False,
        save_output=True,
    )

    ################################
    ##### Use the Pydra system #####
    ################################
    print(f"** Running {pipeline_name} in Pydra")

    # Initialize the Pydra Workflow
    wf = pydra.Workflow(
        name=pipeline_name, input_spec=["config"], config=pipeline_config
    )
    pipeline_outputs = list()

    # Start calling the pipeline stuff
    pydra_step = "pupil_detection_left"
    wf.add(pupil_detection(name=pydra_step, eye="left", config=wf.lzin.config))
    pipeline_outputs.append((pydra_step, wf.pupil_detection_left.lzout.out))

    pydra_step = "pupil_detection_right"
    wf.add(pupil_detection(name=pydra_step, eye="right", config=wf.lzin.config))
    pipeline_outputs.append((pydra_step, wf.pupil_detection_right.lzout.out))

    pydra_step = "calibration_marker_detection"
    wf.add(marker_detection(name=pydra_step, config=wf.lzin.config))
    pipeline_outputs.append((pydra_step, wf.calibration_marker_detection.lzout.out))

    pydra_step = "calibration_marker_filtering"
    wf.add(
        marker_filtering(
            name=pydra_step, marker_fpath=wf.calibration_marker_detection.lzout.out
        )
    )
    pipeline_outputs.append((pydra_step, wf.calibration_marker_filtering.lzout.out))

    pydra_step = "calibrate_left"
    wf.add(
        calibrate(
            name=pydra_step,
            marker_fpath=wf.calibration_marker_filtering.lzout.out,
            pupil_fpath=wf.pupil_detection_left.lzout.out,
        )
    )
    pipeline_outputs.append((pydra_step, wf.calibrate_left.lzout.out))

    pydra_step = "calibrate_right"
    wf.add(
        calibrate(
            name=pydra_step,
            marker_fpath=wf.calibration_marker_filtering.lzout.out,
            pupil_fpath=wf.pupil_detection_right.lzout.out,
        )
    )
    pipeline_outputs.append((pydra_step, wf.calibrate_right.lzout.out))

    pydra_step = "map_gaze_left"
    wf.add(
        map_gaze(
            name=pydra_step,
            calibration_fpath=wf.calibrate_left.lzout.out,
            pupil_fpath=wf.pupil_detection_left.lzout.out,
        )
    )
    pipeline_outputs.append((pydra_step, wf.map_gaze_left.lzout.out))

    pydra_step = "map_gaze_right"
    wf.add(
        map_gaze(
            name=pydra_step,
            calibration_fpath=wf.calibrate_right.lzout.out,
            pupil_fpath=wf.pupil_detection_right.lzout.out,
        )
    )
    pipeline_outputs.append((pydra_step, wf.map_gaze_right.lzout.out))

    pydra_step = "validation_marker_detection"
    wf.add(
        marker_detection(
            name=pydra_step, marker_type="validation", config=wf.lzin.config
        )
    )
    pipeline_outputs.append((pydra_step, wf.validation_marker_detection.lzout.out))

    pydra_step = "validation_marker_filtering"
    wf.add(
        marker_filtering(
            name=pydra_step, marker_fpath=wf.validation_marker_detection.lzout.out
        )
    )
    pipeline_outputs.append((pydra_step, wf.validation_marker_filtering.lzout.out))

    pydra_step = "compute_error_left"
    wf.add(
        compute_error(
            name=pydra_step,
            marker_fpath=wf.validation_marker_filtering.lzout.out,
            gaze_fpath=wf.map_gaze_left.lzout.out,
        )
    )
    pipeline_outputs.append((pydra_step, wf.compute_error_left.lzout.out))

    pydra_step = "compute_error_right"
    wf.add(
        compute_error(
            name=pydra_step,
            marker_fpath=wf.validation_marker_filtering.lzout.out,
            gaze_fpath=wf.map_gaze_right.lzout.out,
        )
    )
    pipeline_outputs.append((pydra_step, wf.compute_error_right.lzout.out))

    # Set the output of the workflow
    wf.set_output(pipeline_outputs)

    # Run the pipeline
    # Can use plugin="serial" to avoid the error, but theoretically that's slower
    # TODO: switch this back on big machines
    # with pydra.Submitter(plugin="cf") as sub:
    with pydra.Submitter(plugin="serial") as sub:
        sub(wf)

    wf.result()
