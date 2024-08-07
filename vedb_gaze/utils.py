# Utilities supporting gaze analysis

import numpy as np


###########################
##### VEDB GAZE UTILS #####
###########################


def unique(seq, idfun=None):
    """Returns only unique values in a list (with order preserved).
    (idfun can be defined to select particular values??)

    Stolen from the internets 11.29.11

    Parameters
    ----------
    seq : TYPE
        Description
    idfun : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    # order preserving
    if idfun is None:

        def idfun(x):
            return x

    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen:
            seen[marker] += 1
            continue
        else:
            seen[marker] = 1
            result.append(item)
    return result, seen


def match_time_points(*data, fn=np.median, window=None):
    """Compute gaze position across matched time points

    Currently selects all gaze points within half a video frame of
    the target time (first data timestamp field) and takes median
    of those values.

    NOTE: This is messy. computing median doesn't work for fields of
    data that are e.g. dictionaries. These must be removed before
    calling this function for now.
    """
    if window is None:
        # Overwite any function argument if window is set to none;
        # this will do nearest-frame resampling
        def fn(x, axis=None):
            return x

    # Timestamps for first input are used as a reference
    reference_time = data[0]["timestamp"]
    # Preallocate output list
    output = []
    # Loop over all subsequent fields of data
    for d in data[1:]:
        t = d["timestamp"].copy()
        new_dict = dict(timestamp=reference_time)
        # Loop over all timestamps in time reference
        for i, frame_time in enumerate(reference_time):
            # Preallocate lists
            if i == 0:
                for k, v in d.items():
                    if k in new_dict:
                        continue
                    shape = v.shape
                    new_dict[k] = np.zeros(
                        (len(reference_time),) + shape[1:], dtype=v.dtype
                    )
            if window is None:
                # Nearest frame selection
                fr = np.argmin(np.abs(t - frame_time))
                time_index = np.zeros_like(t) > 0
                time_index[fr] = True
            else:
                # Selection of all frames within window
                time_index = np.abs(t - frame_time) < window
            # Loop over fields of inputs
            for k, v in d.items():
                if k == "timestamp":
                    continue
                try:
                    frame = fn(v[time_index], axis=0)
                    new_dict[k][i] = frame
                except:
                    # Field does not support indexing of this kind;
                    # This should probably raise a warning at least...
                    pass
        # Remove any keys with all fields deleted
        keys = list(d.keys())
        for k in keys:
            if len(new_dict[k]) == 0:
                _ = new_dict.pop(k)
            else:
                new_dict[k] = np.asarray(new_dict[k])
        output.append(new_dict)
    # Flexible output, depending on number of inputs
    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)


def onoff_from_binary(data, return_duration=True):
    """Converts a binary variable data into onsets, offsets, and optionally durations

    This may yield unexpected behavior if the first value of `data` is true.

    Parameters
    ----------
    data : array-like, 1D
        binary array from which onsets and offsets should be extracted
    return_duration : bool, optional
        Description

    Returns
    -------
    TYPE
        Description

    """
    if data[0]:
        start_value = 1
    else:
        start_value = 0
    data = data.astype(float).copy()

    ddata = np.hstack([[start_value], np.diff(data)])
    (onsets,) = np.nonzero(ddata > 0)
    # print(onsets)
    (offsets,) = np.nonzero(ddata < 0)
    # print(offsets)
    onset_first = onsets[0] < offsets[0]
    len(onsets) == len(offsets)

    on_at_end = False
    on_at_start = False
    if onset_first:
        if len(onsets) > len(offsets):
            offsets = np.hstack([offsets, [-1]])
            on_at_end = True
    else:
        if len(offsets) > len(onsets):
            onsets = np.hstack([-1, offsets])
            on_at_start = True
    onoff = np.vstack([onsets, offsets])
    if return_duration:
        duration = offsets - onsets
        if on_at_end:
            duration[-1] = len(data) - onsets[-1]
        if on_at_start:
            duration[0] = offsets[0] - 0
        onoff = np.vstack([onoff, duration])

    onoff = onoff.T.astype(int)
    return onoff


def onoff_to_binary(onoff, length):
    """Convert (onset, offset) tuples to binary index

    Parameters
    ----------
    onoff : list of tuples
        Each tuple is (onset_index, offset_index, [duration_in_frames]) for some event
    length : total length of output vector
        Scalar value for length of output binary index

    Returns
    -------
    index
        boolean index vector
    """
    index = np.zeros(length,)
    for on, off in onoff[:, :2]:
        index[on:off] = 1
    return index > 0


def filter_list(lst, idx):
    """Convenience function to select items from a list with a binary index"""
    return [x for x, i in zip(lst, idx) if i]


def filter_arraydict(arraydict, idx):
    """Apply the same index to all fields in a dict of arrays"""
    dictlist = arraydict_to_dictlist(arraydict)
    dictlist = filter_list(dictlist, idx)
    out = dictlist_to_arraydict(dictlist)
    return out


def dictlist_to_arraydict(dictlist):
    """Convert from pupil format list of dicts to dict of arrays"""
    dict_fields = list(dictlist[0].keys())
    out = {}
    for df in dict_fields:
        out[df] = np.array([d[df] for d in dictlist])
    return out


def arraydict_to_dictlist(arraydict):
    """Convert from dict of arrays to pupil format list of dicts"""
    dict_fields = list(arraydict.keys())
    first_key = dict_fields[0]
    n = len(arraydict[first_key])
    out = []
    for j in range(n):
        frame_dict = {}
        for k in dict_fields:
            value = arraydict[k][j]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            frame_dict[k] = value
        out.append(frame_dict)
    return out


def stack_arraydicts(*inputs, sort_key=None):
    output = arraydict_to_dictlist(inputs[0])
    for arrdict in inputs[1:]:
        arrlist = arraydict_to_dictlist(arrdict)
        output.extend(arrlist)
    if sort_key is not None:
        output = sorted(output, key=lambda x: x[sort_key])
    # Handle case in which all fields are empty.
    if len(output) > 0:
        output = dictlist_to_arraydict(output)
    else:
        # In degenerate case, return first dict of empty arrays
        # This might be a one-off fix, unclear
        output = inputs[0]
    return output


def get_function(function_name):
    """Load a function to a variable by name

    Parameters
    ----------
    function_name : str
        string name for function (including module)
    """
    if callable(function_name):
        return function_name
    import importlib

    fn_path = function_name.split(".")
    module_name = ".".join(fn_path[:-1])
    fn_name = fn_path[-1]
    module = importlib.import_module(module_name)
    func = getattr(module, fn_name)
    return func


############################
##### VEDB STORE UTILS #####
############################

# These are ripped from vedb_store.utils so that we don't have to rely on that library
# https://github.com/vedb/vedb-store/blob/main/vedb_store/utils.py


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
    ti = (all_time > start_time) & (all_time < end_time)
    (indices,) = np.nonzero(ti)
    start_frame, end_frame = indices[0], indices[-1] + 1
    return start_frame, end_frame


def parse_resolution(res_string):
    """Cleans up resolution string from YAML file"""
    out = res_string.strip("()").split(",")
    out = [int(x) for x in out]
    return out
