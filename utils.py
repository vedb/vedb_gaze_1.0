#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:56:19 2023

@author: mgreene2
"""

import numpy as np
import cv2
import os

def get_frame_indices(start_time, end_time, all_time):
	"""Finds start and end indices for frames that are between `start_time` and `end_time`
	
	Note that `end_frame` returned will be the first frame that occurs after
	end_time, such that some data[start_frame:end_frame] will span the range 
	between `start_time` and `end_time`. 
	Parameters
	----------
	start_time: scalar
		time after which to select frames
	end_time: scalar
		time before which to select frames
	all_time: array-like
		full array of timestamps for data into which to index.
	"""
	ti = (all_time > (all_time[0] + start_time)) & (all_time < (all_time[0] + end_time))
	indices, = np.nonzero(ti)
	start_frame, end_frame = indices[0], indices[-1] + 1
	return start_frame, end_frame 
	
def load_mp4(fpath, 
            frames=(0,100), 
            size=None, 
            crop_size=None, 
            center=None, 
            pad_value=None, 
            tmpdir='/tmp/mp4cache/', 
            color='rgb', 
            loader='opencv'):
    """
    Parameters
    ----------
    fpath : string
        path to file to load
    frames : tuple
        (first, last) frame to be loaded. If not specified, attempts to load 
        first 100 frames
    size : tuple or scalar float or None
        desired final size of output, (vertical_dim x horizontal_dim), or 
    crop_size : tuple or scalar float or None
        size of image to crop around center. This can be further resized after
        cropping with the `size` parameter, which governs final size 
    center : array-like
        list or array or tuple for (x,y) coordinates around which to 
        crop each frame. 
    tmpdir : string path
        path to download mp4 file if it initially exists in a remote location
    loader : string
        'opencv' or 'imageio' ImageIO is slower, clearer what it's doing...
    color : string
        'rgb', 'bgr', or 'gray'
        'gray' converts to LAB space, keeps luminance channel, returns values from 0-100
    Notes
    -----
    Defaults to loading first 100 frames. 
    """

    path, fname = os.path.split(fpath)
    file_name = fpath
    
    # Prep for resizing if necessary
    vid = cv2.VideoCapture(file_name)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    size = (int(height), int(width))
    if isinstance(size, (tuple, list)):
                    resize_fn = lambda im: cv2.resize(im, size[::-1], interpolation=cv2.INTER_AREA)
    
    # Allow output to just be size of crop
    if (size is not None) and (crop_size is not None):
        size = crop_size
    #  Handle color mode
    if color == 'rgb':
        color_fn =  lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif color=='gray':
            color_fn =  lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[:, :, 0]
    elif color=='bgr':
        color_fn = lambda im: im
    else:
        raise ValueError('Unknown color conversion!')
    
    # Preallocate image to read
    n_frames = frames[1] - frames[0]
    if (center is None) or np.ndim(center) == 1:
        center = [center] * n_frames
    if len(center) != n_frames:
        raise ValueError('`center` must be a single tuple or the same number of frames to be loaded!')
    if isinstance(size, (list, tuple)):
        imdims = size
    else:
        orig_imdims = np.array(var_size(file_name)[1:3])
        imdims = np.ceil(size * orig_imdims).astype(np.int)
    if color=='gray':
        output_dims = imdims
    else:
        output_dims = (imdims[0], imdims[1], 3)
    imstack = np.zeros((n_frames, *output_dims), dtype=np.uint8)
    # Load from local file; with clause should correctly close ffmpeg instance
    if frames is None:
        frames = (0, int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
    vid.set(1,frames[0])
    
    # Call resizing function on each frame individually to
    # minimize memory overhead
    for i, fr in enumerate(range(*frames)):
        tmp = vid.read()[1]
        if center[i] is not None:
            tmp = crop_frame(tmp, 
                                center=center[i], 
                                size=crop_size, 
                                pad_value=pad_value)
        imstack[i] = color_fn(resize_fn(tmp))

    return imstack
