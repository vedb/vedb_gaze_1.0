# vedb_gaze_1.0
Processing of gaze for visual experience dataset. This code is a light-dependency version of the code in [vedb-gaze](http://github.com/vedb/vedb-gaze). 

This code performs the following operations: 

1. Estimate pupil position from eye video frames with PupilLabs `pupildetectors` library
2. Detect gaze calibration markers (concentric circles) in world video frames
3. Detect gaze validation markers (checkerboards) in world video frames
4. Filter detected calibration and validation markers to remove spurious detections
5. Estimate calibration (mapping from gaze video to world video coordinates) with 2D polynomial mapping as in PupilLabs
6. Map gaze to normalized world video coordinates (0-1 in vertical and horizontal directions)
7. Estimate error on detected validation markers

The pipeline code is designed for use with data from the Visual Experience Dataset, which can be found on [Databrary](http://nyu.databrary.org/volumes/1612) and the [Open Science Foundation](https://osf.io/2gdkb/)

Indvidual functions may be used separately, e.g. vedb_gaze.pupil_detection_pl.plabs_detect_pupil can be used with any eye video data to detect pupils, and any world video data to detect markers. Data structures are compatible with PupilLabs PupilCapture software. 

Instructions for setting up the code to run and some practical examples are included below. 

# Set up environment
To install the dependencies for this code in anaconda, please go to this directory in your terminal app and call the following with your anaconda base environment active:

`conda env create -f environment.yml`

This will create an anaconda environment called vedb_analysis_1.0. To activate the environment, call: 

`conda activate vedb_analysis_1.0`

On Linux systems, you may also need to install the g++ compiler for some dependencies.

This code has not been tested on windows systems.

# Examples

## 1. Run full pipeline

To analyze a full session, this code expects at least the following files to be present: 

```
2021_02_12_13_51_20/
├── eye0_blur.mp4
├── eye0_timestamps.npy
├── eye1_blur.mp4
├── eye1_timestamps.npy
├── marker_times.yaml
├── worldPrivate.mp4
└── world_timestamps.npy
```

Note that for VEDB sessions, this includes files stored on both [Databrary](http://nyu.databrary.org/volumes/1612) and the [Open Science Foundation](https://osf.io/2gdkb/). Files from both sources must be copied to the same folder. To run analysis for a session, call: 

`python extract_gaze_pipeline.py -f /path/to/my/session`


## 2. Run a single step (here, detect pupil positions) within a python session

The following can be called to run pupil detection on a given eye video. Results are returned as a dict.

```python
import vedb_gaze
video_file = '/path/to/vedb/session/eye0.mp4'
timestamp_file = '/path/to/vedb/session/eye0_timestamp.npy'
out = vedb_gaze.pupil_detection_pl.plabs_detect_pupil(eye_file, timestamp_file=timestamp_file, start_frame=0, end_frame=2000, id=0, )
```

# Reference
Greene, M.R., Balas, B.J., Lecroart, M.D., MacNeilage, P.R., Hart, J.A., Binaee, K., Hausamann, P.A., Mezile, R., Shankar, B., Sinnott, C.B., Capurro, K., Halow, S., Howe, H., Josyula, M., Li, A., Mieses, A., Mohamed, A., Nudnou, I., Parkhill, E., Riley, P., Schmidt, B., Shinkle, M.W., Si, W., Szekely, B, Torres, J.M., Weissmann, E. (2024) The Visual Experience Dataset: Over 200 Recorded Hours of Integrated Eye Movement, Odometry, and Egocentric Video. ArXiv.
