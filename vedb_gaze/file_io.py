# Loading / saving of files (as distinct from classes, which is higher-level)

# Imports (vm_tools)
import os
import six
import time
import glob
import h5py
import json
import uuid
import shutil
import imageio
import tempfile
import warnings
import inspect
import functools
import subprocess
import collections
import numpy as np
from PIL import Image
from scipy.io import loadmat
from matplotlib.pyplot import imread as _imread

# from . import options

# Soft imports for obscure or heavy modules
try:
    # tqdm for fancy progress tracking
    from tqdm import tqdm
except ImportError:

    def tqdm(x):
        return x


try:
    # scikit-image for image resizing
    from skimage import transform as skt

    skimage_available = True
except ImportError:
    skimage_available = False

try:
    # Gallant lab cotton candy (cloud file access)
    import cottoncandy as cc
    from botocore.client import Config

    default_bucket = cc.default_bucket
    # Longer time-outs for read
    botoconfig = Config(connect_timeout=50, read_timeout=10 * 60)  # 10 mins
except:
    # TODO: put option to fail silently into config file
    print("Failed to import cottoncandy - no cloud interfaces available!")
    botoconfig = None
    default_bucket = None  # This will fail... but so it goes.
    botoconfig = None

try:
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    torch_available = True
except:
    torch_available = False

try:
    import cv2

    opencv_available = True
except ImportError:
    opencv_available = False

try:
    import msgpack

    msgpack_available = True
except ImportError:
    msgpack_available = False

# Parameters
HDF_EXTENSIONS = (".hdf", ".hf", ".hdf5", ".h5", ".hf5")

# functions
def load_image(fpath, mode="RGB", loader="matplotlib"):
    """Dead simple imread with matplotlib (only) for now. 

    A placeholder for a more useful layer of abstraction.
    """
    if loader == "matplotlib":
        im = _imread(fpath)
    elif loader == "PIL":
        pil_image = Image.open(fpath)
        if pil_image.mode == "RGB":
            n_channels = 3
        elif pil_image.mode == "RGBA":
            n_channels = 4
        im = np.array(pil_image.getdata()).reshape(
            pil_image.size[1], pil_image.size[0], n_channels
        )
    if loader == "opencv":
        im = cv2.imread(fpath)
        if mode in ["RGB", "RGBA"]:
            im = im[..., ::-1]
    if mode == "RGB":
        if np.ndim(im) == 3 and im.shape[2] == 4:
            # Clip alpha channel
            # Note: there are other options here. Image may e.g. be
            # black where alpha is clear; better may be to provide
            # an underlay for images with active alpha channels
            im = im[:, :, :3]
        elif np.ndim(im) == 3 and im.shape[2] == 3:
            # Fine, do nothing.
            pass
        else:
            raise Exception("2D only image or something error error no good handle me")
    elif mode == "RGBA":
        # Need to add alpha channel if it doesn't exist.
        raise NotImplementedError("RGBA image loading not ready yet.")
    return im


class VideoCapture(object):
    """Tweak of opencv VideoCapture to allow working with "with" statements
    per https://github.com/skvark/opencv-python/issues/205
    """

    def __init__(self, device_num):
        self.VideoObj = cv2.VideoCapture(device_num)

    def __enter__(self):
        return self.VideoObj

    def __exit__(self, type, value, traceback):
        self.VideoObj.release()


def crop_frame(frame, center, size=(512, 512), pad_value=None):
    """Crop an arrray to a size around a particular center point
    
    Parameters
    ----------
    frame : array
        image to be cropped
    center : array-like
        list, tuple, or array with (x, y) center coordinate in normalized
        (0-1) coordinates
    size : array-like
        size (vertical, horizontal) of image
    pad_value : scalar 
        value to use for image padding if desired cropped region goes 
        outside of image area (function always returns same size array)
    
    Returns
    -------
    cropped_image : array
        cropped array result
    """
    # Handle errors up front:
    if np.any((center > 1) | (center < 0)):
        return np.zeros(size, dtype=frame.dtype)
    vdim, hdim = size
    frame_vdim, frame_hdim = frame.shape[:2]
    center = np.array(center) * np.array([frame_hdim, frame_vdim])
    center = np.round(center).astype(np.int)
    vst, vfin = center[1] - np.int(vdim / 2), center[1] + np.int(vdim / 2)
    hst, hfin = center[0] - np.int(hdim / 2), center[0] + np.int(hdim / 2)
    # overflow
    vunder = -np.minimum(vst, 0)
    vover = np.maximum(vfin - frame_vdim, 0)
    hunder = -np.minimum(hst, 0)
    hover = np.maximum(hfin - frame_hdim, 0)
    # Correct indices
    vst = np.maximum(vst, 0)
    vfin = np.minimum(vfin, frame_vdim)
    hst = np.maximum(hst, 0)
    hfin = np.minimum(hfin, frame_hdim)
    # Crop
    region = frame[vst:vfin, hst:hfin]
    if pad_value is None:
        if region.dtype == np.uint8:
            constant_value = 0
        else:
            constant_value = np.nan
    else:
        constant_value = pad_value
    # Pad
    if np.ndim(frame) == 3:
        out = np.pad(
            region,
            [[vunder, vover], [hunder, hover], [0, 0]],
            mode="constant",
            constant_values=constant_value,
        )
    elif np.ndim(frame) == 2:
        out = np.pad(
            region,
            [[vunder, vover], [hunder, hover],],
            mode="constant",
            constant_values=constant_value,
        )
    return out


def load_mp4(
    fpath,
    frames=(0, 100),
    size=None,
    crop_size=None,
    center=None,
    pad_value=None,
    tmpdir="/tmp/mp4cache/",
    color="rgb",
    loader="opencv",
):
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
    # Check for cloud path; if so, use cottoncandy for s3 access
    path, fname = os.path.split(fpath)
    bucket, virtual_dirs = cloud_bucket_check(path)
    if bucket is None:
        # No cottoncandy
        file_name = fpath
    else:
        # Cottoncandy. Download from remote server.
        cci = get_interface(bucket)
        file_name = os.path.join(tmpdir, fname)
        if not os.path.exists(file_name):
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)
            cci.download_to_file(os.path.join(virtual_dirs, fname), file_name)
    interp = cv2.INTER_AREA  # TO DO: make it clearer what this is doing
    # (bilinear, cubic spline, ...?)
    # Prep for resizing if necessary
    if size is None:
        resize_fn = lambda im: im
        if loader == "opencv":
            with VideoCapture(file_name) as vid:
                width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
                height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
                size = (int(height), int(width))
        else:
            with imageio.get_reader(file_name) as vid:
                size = vid.get_meta_data()["source_size"]
    else:
        if loader == "imageio":
            if skimage_available:
                if isinstance(size, (tuple, list)):
                    resize_fn = lambda im: skt.resize(
                        im, size, anti_aliasing=True, order=3, preserve_range=-True
                    ).astype(im.dtype)
                else:
                    # float provided
                    resize_fn = lambda im: skt.rescale(
                        im, size, anti_aliasing=True, order=3, preserve_range=-True
                    ).astype(im.dtype)
            else:
                raise ImportError(
                    "Please install scikit-image to be able to resize videos at load"
                )
        elif loader == "opencv":
            if opencv_available:
                if isinstance(size, (tuple, list)):
                    resize_fn = lambda im: cv2.resize(
                        im, size[::-1], interpolation=interp
                    )
                else:
                    resize_fn = lambda im: cv2.resize(
                        im, None, fx=size, fy=size, interpolation=interp
                    )
            else:
                raise ImportError(
                    "Please install opencv to be able to resize videos at load"
                )
    # Allow output to just be size of crop
    if (size is not None) and (crop_size is not None):
        size = crop_size
    #  Handle color mode
    if loader == "opencv":
        if color == "rgb":
            color_fn = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif color == "gray":
            color_fn = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[:, :, 0]
        elif color == "bgr":
            color_fn = lambda im: im
        else:
            raise ValueError("Unknown color conversion!")
    elif loader == "imageio":
        if color == "rgb":
            color_fn = lambda im: im
        else:
            NotImplementedError("Color conversions with skimage not implemented yet!")

    # Preallocate image to read
    n_frames = frames[1] - frames[0]
    if (center is None) or np.ndim(center) == 1:
        center = [center] * n_frames
    if len(center) != n_frames:
        raise ValueError(
            "`center` must be a single tuple or the same number of frames to be loaded!"
        )
    if isinstance(size, (list, tuple)):
        imdims = size
    else:
        orig_imdims = np.array(var_size(file_name)[1:3])
        imdims = np.ceil(size * orig_imdims).astype(np.int)
    if color == "gray":
        output_dims = imdims
    else:
        output_dims = (imdims[0], imdims[1], 3)
    imstack = np.zeros((n_frames, *output_dims), dtype=np.uint8)
    # Load from local file; with clause should correctly close ffmpeg instance
    if loader == "opencv":
        with VideoCapture(file_name) as vid:
            if frames is None:
                frames = (0, int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
            vid.set(1, frames[0])
            # Call resizing function on each frame individually to
            # minimize memory overhead
            for i, fr in enumerate(range(*frames)):
                tmp = vid.read()[1]
                if center[i] is not None:
                    tmp = crop_frame(
                        tmp, center=center[i], size=crop_size, pad_value=pad_value
                    )
                imstack[i] = color_fn(resize_fn(tmp))
    elif loader == "imageio":
        with imageio.get_reader(file_name, "ffmpeg") as vid:
            if frames is None:
                frames = (0, vid.count_frames())
            # Call resizing function on each frame individually to
            # minimize memory overhead
            for i, fr in enumerate(range(*frames)):
                if i == 0:
                    tmp = vid.get_data(fr)
                else:
                    tmp = vid.get_next_data()
                if center[i] is not None:
                    tmp = crop_frame(
                        tmp, center=center[i], size=crop_size, pad_value=pad_value
                    )
                imstack[i] = color_fn(resize_fn(tmp))

    return imstack


def write_movie_from_frames(files, sname, fps=24, progress_bar=tqdm, **kwargs):
    """
    files is a list of files (sort it! )
    sname is the name of the movie to be written
    movie_size is (ht, width) e.g. (600,800)
    fps is frames per second, ideally integer
    """
    im0 = load_image(str(files[0]), loader="opencv")
    movie_size = im0.shape[:2]  # [::-1]
    vid = VideoEncoderFFMPEG(sname, movie_size, fps, **kwargs)
    for fnm in progress_bar(files):
        im = load_image(str(fnm), loader="opencv")
        vid.write(im)
    vid.stop()


def load_msgpack(fpath, idx=None):
    """Load a list of dictionaries from a msgpack file

    Parameters
    ----------
    fpath : string
        path to file
    idx : tuple
        NOT FUNCTIONAL YET. Does nothing. Intending to make this indices into the list of dicts.

    Returns
    -------
    list of contents of file, whatever those are. Generally, dicts. 

    Notes
    -----
    Intended for use with pupil labs data (.pldata files). Unclear if this will generalize to other 
    msgpack files.

    See https://stackoverflow.com/questions/43442194/how-do-i-read-and-write-with-msgpack for 
    basics of reading and writing to msgpack; 
    See https://stackoverflow.com/questions/42907315/unpacking-msgpack-from-respond-in-python
    for notes on unpacking in chunks.
    """
    data = []
    if idx is None:
        idx = (-1, np.inf)
    with open(fpath, "rb") as fh:
        for i, (topic, payload) in enumerate(
            msgpack.Unpacker(fh, raw=False, use_list=False)
        ):
            if (i >= idx[0]) and i < idx[-1]:
                data.append(msgpack.unpackb(payload, raw=False))

    return data


def nifti_from_volume(vol, inputnii, sname=None):
    import nibabel

    outnii = nibabel.Nifti1Image(vol, np.eye(4))
    # Update transforms in header w/ old transform (unchanged by this step)
    outnii.set_sform(inputnii.get_sform())
    outnii.set_qform(inputnii.get_qform())
    outnii.update_header()
    outnii.set_data_dtype(64)
    if not sname is None:
        outnii.to_filename(sname)
    return outnii


def get_interface(bucket_name=default_bucket, verbose=False, config=botoconfig):
    """Wrapper to manage syntax diffs btw google drive & s3 interfaces

    Unclear if this function is worth the trouble.
    """
    if bucket_name == "gdrive":
        cci = cc.get_interface(backend=bucket_name, verbose=verbose)
    else:
        cci = cc.get_interface(bucket_name=bucket_name, verbose=verbose, config=config)
    return cci


def cloud_bucket_check(path):
    """Test whether a path refers to an amazon S3-style cloud object

    Parse path into bucket + virtual directories if it is a cloud path
    
    Parameters
    ----------
    path : str
        To specify a cloud path, `path` should be of the form:
        'cloud:<bucket>:<virtual_dirs>'
        OR 
        '/s3/<bucket>/<virtual_dirs>'
        OR
        's3://<bucket>/<virtual_dirs>'
        e.g. 
        'cloud:mybucket:my/data/lives/here/'
        OR
        '/s3/mybucket/my/data/lives/here/'
    """
    if path[:6] == "cloud:":
        fd = path.split(":")
        if len(fd) == 3:
            _, bucket, virtual_dirs = fd
        elif len(fd) == 2:
            _, bucket = fd
            virtual_dirs = ""
        else:
            raise Exception(
                "Bad cloud path specified! should be cloud:<bucket>:<virtual_dirs>"
            )
        return bucket, virtual_dirs
    elif path[:4] == "/s3/":
        fd = path.split("/")
        if len(fd) == 3:
            _, _, bucket = fd
            virtual_dirs = ""
        elif len(fd) == 4:
            _, _, bucket, virtual_dirs = fd
        else:
            _, _, bucket = fd[:3]
            virtual_dirs = "/".join(fd[3:])
        return bucket, virtual_dirs
    elif path[:5] == "s3://":
        fd = path[5:].split("/")
        bucket = fd[0]
        virtual_dirs = "/".join(fd[1:])
        return bucket, virtual_dirs

    else:
        return None, path


def fexists(fpath, variable_name=None):
    """Check whether a file exists, in a file system or on the cloud

    If you are checking on a cloud file, path should be of the form:
    cloud:<bucket>:<virtual_dirs>,
    e.g. 
    cloud:mybucket:my/data/lives/here/

    variable_name is only for cloud files. On the cloud (so far), you can only 
    check whether a given array - a single object - exists, not the
    hdf-like virtual path that groups multiple objects.
    """
    path, fname = os.path.split(fpath)
    bucket, fpath = cloud_bucket_check(path)
    if bucket is None:
        fpath = os.path.join(path, fname)
        file_exists = os.path.exists(fpath)
        if variable_name is None:
            return file_exists
        else:
            if not file_exists:
                raise ValueError(
                    "Cannot check for variable_name, parent file does not exist"
                )
            with warnings.catch_warnings():
                # Ignore bullshit h5py/tables warning
                warnings.simplefilter("ignore")
                with h5py.File(fpath, "r") as hf:
                    var_exists = variable_name in hf
            return var_exists
    else:
        cloudi = get_interface(bucket_name=bucket, verbose=False, config=botoconfig)
        oname = os.path.join(fpath, fname)
        if variable_name is None:
            files = cloudi.glob(oname)
            file_exists = len(files) > 0
        else:
            array_name = os.path.join(oname, variable_name)
            file_exists = cloudi.exists_object(array_name)
        return file_exists


def file_array_keys(fpath):
    """Get keys for variable stored in a file

    Does NOT support cloud arrays yet.
    """
    fnm, ext = os.path.splitext(fpath)
    if ext in (".mp4",):
        return None
    elif ext in HDF_EXTENSIONS:
        with h5py.File(fpath, mode="r") as hf:
            return list(hf.keys())
    elif ext in (".mat"):
        try:
            # Try hdf first
            with h5py.File(fpath, mode="r") as hf:
                return list(hf.keys())
        except:
            # Try .mat second (loads all variables, lame)
            d = loadmat(fpath)
            return list(d.keys())
    else:
        raise ValueError("Untenable file type")


def var_size(fpath, variable_name=None, cloudi=None):
    """"""
    path, fname = os.path.split(fpath)
    bucket, path = cloud_bucket_check(path)
    if bucket is None:
        fstr, ext = os.path.splitext(fname)
        if ext in HDF_EXTENSIONS:
            with h5py.File(fpath, mode="r") as hf:
                return hf[variable_name].shape
        elif ext in ".mp4":
            if opencv_available:
                with VideoCapture(fpath) as vid:
                    vdim = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    hdim = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                    n_frames_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                return [n_frames_total, vdim, hdim, 3]
            else:
                vid = imageio.get_reader(fpath, "ffmpeg")
                meta = vid.get_meta_data()
                x, y = meta["size"]
                # precise, slow:
                nf = vid.count_frames()
                # imprecise (?), fast:
                # nf = np.round(meta['duration'] * meta['fps']).astype(np.int)
                return [nf, y, x, 3]

        elif ext in ".npy":
            arr_memmap = np.load(fpath, mmap_mode="r")
            return arr_memmap.shape
        else:
            raise ValueError("Only usable for hdf, mp4, and npy files for now.")


# def get_array_size(fname, axis=0):
#     """Get total number of frames (or other quantity) in file"""
#     pass


def lsfiles(prefix, cloudi=None, keep_prefix=False):
    """List all files with particular prefix"""
    bucket, fpath = cloud_bucket_check(prefix)
    if bucket is None:
        # Standard file system
        return glob.glob(fpath)
    else:
        if cloudi is None:
            cloudi = get_interface(bucket_name=bucket, verbose=False)
        if bucket == "gdrive":
            path, fname = os.path.split(fpath)
            if "*" in path:
                raise Exception(
                    "glob syntax for multiple files does not work with google drive!"
                )
            files = cloudi.lsdir(path + os.path.sep)
            if fname != "":
                if "*" in fname:
                    # Crude
                    idx = fname.find("*")
                    files_a = [f for f in files if f[:idx] == fname[:idx]]
                    print(files_a)
                    if fname[idx + 1 :] != "":
                        n_end = len(fname) - idx - 1
                        files_b = [f for f in files if (f[-n_end:] == fname[-n_end:])]
                        print(files_b)
                        files = [f for f in files_a if f in files_b]
                    else:
                        files = files_a
            files = [os.path.join(path, f) for f in files]
        else:
            files = cloudi.glob(fpath)
        if keep_prefix:
            if bucket == "gdrive":
                fprefix = "~/"
            else:
                fprefix = "/s3/"
            files = [os.path.join(fprefix, bucket, f) for f in files]
        return sorted(files)


def cpfile(infile, outfile, cloudi=None, overwrite=False, tmp_file="tmp"):
    """copy a file

    Either from: 
    file to file
    file to google drive
    file to s3
    s3 to file
    s3 to google drive (?)
    s3 to s3 (?)
    googledrive to file
    google drive to google drive (?)
    google drive to s3 (?)
    """

    inbucket, infile_ = cloud_bucket_check(infile)
    outbucket, outfile_ = cloud_bucket_check(outfile)
    # Generic check if outfile exists
    if fexists(outfile) and not overwrite:
        raise Exception("Refusing to over-write extant file %s" % outfile)
    # Handle tmp files
    if tmp_file == "tmp":
        tmp_file = tempfile.mktemp(suffix=".hdf", dir="/tmp/")
    else:
        raise ValueError("value for tmp_file not supported!")
    # Switch over file transfer / copy types
    if (inbucket is None) and (outbucket is None):
        # file to file
        shutil.copy(infile, outfile)
    elif (inbucket is None) and (outbucket is not None):
        # file to google drive / s3
        raise NotImplementedError("Requested file copy not implemented yet")
    elif (inbucket is not None) and (outbucket is None):
        # google drive / s3 to file
        if os.path.splitext(outfile)[-1] not in (".hdf", ".hf5"):
            raise ValueError(
                "For now, file to copy cloud files to must be an hdf file!"
            )
        var_names = lsfiles(infile)
        with h5py.File(outfile, mode="w") as hf:
            for full_obj in var_names:
                cfpath, key = os.path.split(full_obj)
                hf[key] = load_array(infile, key)

    elif inbucket is not None and (outbucket is not None):
        # google drive / s3 to google drive / s3
        if inbucket == outbucket:
            if cloudi is None:
                cloudi = get_interface(bucket_name=inbucket, verbose=False)
            cloudi.cp(infile, outfile, overwrite=overwrite)
            return
        # (always?) copy to local file (or temp directory, or in-memory file)
        # then upload in second step to requested resource
        else:
            raise NotImplementedError("Only within-bucket copying works so far!")
    return


def load_array(
    fpath, variable_name=None, idx=None, random_wait=0, cache_dir=None, **kwargs
):
    """Load named variable from file (on cloud or hdf file)
    TO DO: huge arrays? Indices? 
    
    Parameters
    ----------
    fpath : str
        full file name to load
    variable_name : str
        variable name to load from file (if applicable; ignored for movies)
    idx : tuple
        2-tuple, start and end indices to pull for array. ONLY selects
        contiguous sets of indices / frames for now. `None` (default) 
        loads full array.
    random_wait : scalar
        How long (in seconds) to wait before loading. Here to avoid over-
        stressing system / network by too many simultaneous read requests.
    kwargs : passed to load_mp4 (and potentially other future functions.)
    """
    path, fname = os.path.split(str(fpath))
    bucket, full_path = cloud_bucket_check(path)
    # Check for cache dir for quicker loading
    if cache_dir is not None:
        cache_fpath = os.path.join(cache_dir, fname)
        if not os.path.exists(cache_fpath):
            print("Copying file to cache directory {}".format(cache_dir))
            cpfile(os.path.join(full_path, fname), cache_fpath)
        path = cache_dir
        fpath = cache_fpath
    # Allow optional random wait to avoid synchronous file reads
    if random_wait > 0:
        wait = np.random.rand() * random_wait
        print("Waiting %0.2f seconds to load files..." % wait)
        time.sleep(wait)
    if bucket is None:
        # Parse file type
        fn, ext = os.path.splitext(fname)
        if ext in HDF_EXTENSIONS:
            out = _load_hdf_array(fpath, variable_name=variable_name, idx=idx)
        elif ext in (".mat",):
            out = _load_mat_array(fpath, variable_name=variable_name, idx=idx)
        elif ext in (".mp4",):
            out = load_mp4(fpath, frames=idx, **kwargs)
        elif ext in (".npy",):
            # Assume loading whole thing is not going to kill memory if
            # we're loading an npy file; bigger arrays should be stored
            # as HDFs or some format that allows partial load
            out = np.load(fpath)
            if idx is not None:
                out = out[idx[0] : idx[1]]

    else:
        cloudi = get_interface(bucket_name=bucket, verbose=False, config=botoconfig)
        oname = os.path.join(full_path, fname, variable_name)
        out = cloudi.download_raw_array(oname)
        if out.shape == ():
            # Map to int or float or whatever
            out = out.reshape(1,)[0]
    return out


def _load_hdf_array(fpath, variable_name=None, idx=None):
    """Load array from hdf file

    Parameters
    ----------
    fpath : str
        file path
    variable_name : str
        variable name to load
    idx : tuple
        (start_index, end_index) to load - only works on FIRST DIMENSION for now.
    """
    if variable_name is None:
        # TODO: soften this? If only one variable exists in file, load that?
        raise ValueError("variable_name must be specified for hdf files")
    with warnings.catch_warnings():
        # Ignore bullshit h5py/tables warning
        warnings.simplefilter("ignore")
        with h5py.File(fpath, "r") as hf:
            # if not variable_name in hf: # This raises very annoying warnings, thus it's off for now
            #    raise ValueError('array "%s" not found in %s!'%(variable_name, fpath))
            if idx is None:
                out = hf[variable_name][:]
            else:
                st, fin = idx
                out = hf[variable_name][st:fin]
    return out


def _load_mat_array(fpath, variable_name=None, idx=None):
    """Load array from matlab .mat file

    TODO: idx
    """
    if variable_name is None:
        # TODO: soften this? If only one variable exists in file, load that?
        raise ValueError("variable_name must be specified for hdf files")
    try:
        # Maybe it's a just a .mat file
        d = loadmat(fpath)
        if not variable_name in d:
            raise ValueError('array "%s" not found in %s!' % (variable_name, fpath))
        return d[variable_name]
    except:  # NotImplementedError:
        # Note: Better to catch a specific error here, but it seems the error for loadmat has changed.
        # Now catching a generic error instead.
        return _load_hdf_array(fpath, variable_name=variable_name, idx=idx)


def save_arrays(
    fpath, fname=None, meta=None, acl="public-read", compression=True, **named_vars
):
    """"Layer of abstraction for saving files.

    fpath : string
        Can be a simple file path (/path/to/my/file.hdf). Alternately, if `path` 
        begins with cloud:<bucket name>:virtual/path/, it is assumed that you 
        want to save arrays in s3 cloud storage. This calls Anwar Nunez' 
        cottoncandy module to upload arrays to S3 storage. 
    fname : string
        file name. Separate from path because I find that cleaner. So there.
    meta : dict or None
        Any meta-data to be stored with arrays. Converted to json, so must be json-
        serializable.
    acl : string
        Only applicable if you are storing to cloud. Specifies s3 permissions for 
        object file uploaded.
    compression : str
        compression string for uploading raw arrays in cottoncandy. "True" defaults
        to different things for local (hdf) arrays and cloud arrays: hdf, 'gzip'; 
        cloud, 'Zstd'
    named_vars : keyword args that specify named arrays to be stored
    
    """
    if fname is not None:
        warnings.warn(
            "Deprecated usage! please use a single fpath input to `save_arrays`"
        )
        fpath = os.path.join(fpath, fname)
    pp, fname = os.path.split(fpath)
    fstub, ext = os.path.splitext(fname)
    bucket, fp = cloud_bucket_check(pp)
    if bucket is None:
        if ext in (".hdf", ".hf5", "hf"):
            # HDF5 file
            if compression is True:
                compression = "gzip"
            elif compression is False:
                compression = None
            _save_arrays_hdf(fpath, meta=meta, compression=compression, **named_vars)
        elif ext in (".npz",):
            np.savez(fpath, **named_vars)
    else:
        if compression is True:
            compression = "Zstd"
        elif compression is False:
            compression = None
        oname = os.path.join(fp, fname)
        _save_arrays_cloud(
            bucket, oname, meta=meta, acl=acl, compression=compression, **named_vars
        )
    return


def _save_arrays_hdf(
    fpath, meta=None, compression="gzip", compression_arg=None, fmode="w", **arrays
):
    """Save arrays to hdf file.
    
    Parameters
    ----------
    fpath : string
        hdf file path
    arrays : dict of key, array pairs
        named arrays to be stored. If a string is provided in place of an array value,
        it is assumed that the array was too big to fit in memory and was stored in 
        another file, which must be copied.
    compression : str, None, or bool
        True for default compression ('gzip'); if string, must be 'gzip' or 'lzf'
    compression_arg : int
        amount of compression (for compression='gzip' only). [0-9], default (in h5py) is 4 

    Notes
    -----
    props (docdict) storage...
    mask storage

    """

    with h5py.File(fpath, mode=fmode) as hf:
        for k, v in arrays.items():
            # Check for large file stored to disk
            if type(v) in six.string_types and os.path.exists(v):
                with h5py.File(v) as hftmp:
                    # Copy dataset from one file to another. should circumvent RAM limits.
                    h5py.h5o.copy(hftmp.id, d, hf.id, d)
                # Remove temporary large file
                os.unlink(v)
            else:
                # hf[k] = v
                if compression_arg is not None:
                    copts = dict(compression_opts=compression_arg)
                else:
                    copts = {}
                hf.create_dataset(k, data=v, compression=compression, **copts)
        if not meta is None:
            # store meta-data
            hf["meta_data"] = json.dumps(meta)


def _save_arrays_cloud(
    bucket, fname, meta=None, acl="public-read", compression="Zstd", **arrays
):
    """Save arrays to hdf file.
    
    Parameters
    ----------
    bucket : string
        bucket name in cloud
    fname : string
        object name in cloud (should end with .hdf, per cotton_candy conventions)
    meta : dict or None
        meta-data for arrays
    acl : string
        access control list (file permissions) for s3 object() created
    arrays : dict of key, array pairs
        named arrays to be stored. 

    Notes
    -----
    props (docdict) storage...
    mask storage
    [what do we do about large arrays? dask arrays??]
    """
    cloudi = get_interface(bucket_name=bucket, verbose=False, config=botoconfig)
    cloudi.dict2cloud(fname, arrays, compression=compression, acl=acl)
    # dask array for huge files?
    if not meta is None:
        fnm, ext = os.path.splitext(fname)
        cloudi.upload_json("".join([fnm, ".json"]), meta)
    return


def save_dict(fpath, tosave, mode="json"):
    """Save a dict to a file
    
    Parameters
    ----------

    Currently `mode` can only be 'json', but there are ambitions to change this to 
    include pickle or fancy versions of pickle
    
    Not recommended to have arrays as part of your dict; use other functions for that
    """
    pp, fname = os.path.split(fpath)
    bucket, fp = cloud_bucket_check(pp)
    oname = os.path.join(pp, fname)
    if bucket is None:
        # save json file
        if mode == "json":
            json.dump(tosave, open(oname, mode="w"))
        elif mode == "yaml":
            pass
        else:
            raise NotImplementedError(
                "Only mode='json' and mode='yaml' work so far for save_dict()!"
            )
    else:
        cloudi = get_interface(bucket_name=bucket, verbose=False, config=botoconfig)
        cloudi.upload_json(oname, tosave)


def load_dict(fpath, fname=None, mode="json"):
    """Load a dictionary from a saved file. For now, only 'json'"""
    if fname is not None:
        warnings.warn(
            "Deprecated usage! please use a single fpath input to `load_dict`"
        )
        fpath = os.path.join(fpath, fname)
    path, fname = os.path.split(fpath)
    bucket, fpath = cloud_bucket_check(path)
    oname = os.path.join(fpath, fname)
    if bucket is None:
        # save json file
        if mode == "json":
            out = json.load(open(oname, mode="r"))
        else:
            raise NotImplementedError("Only mode='json' works so far for load_dict()!")
    else:
        cloudi = get_interface(bucket_name=bucket, verbose=False, config=botoconfig)
        out = cloudi.download_json(oname)
    return out


def delete(fpath, fname=None, key=None, verbose=False):
    """Delete a file or cloud object/directory of objects

    if no `key` argument is provided, all arrays in fname group are deleted
    (*this isn't working yet for regular hdf files)
    """
    if fname is not None:
        warnings.warn("Deprecated usage! please use a single fpath input to `delete`")
        fpath = os.path.join(fpath, fname)
    if not fexists(fpath):
        raise Exception("File %s not found!" % os.path.join(path, fname))
    path, fname = os.path.split(fpath)
    bucket, fpath = cloud_bucket_check(path)
    if bucket is None:
        # File system. Incorporate deletion of individual keys.
        os.unlink(os.path.join(path, fname))
    else:
        # Cloud
        cloudi = get_interface(bucket_name=bucket, verbose=verbose, config=botoconfig)
        object_stem = os.path.join(fpath, fname)
        if key is None:
            if object_stem[-1] == "*":
                # Asterisk implies remove all
                cloudi.rm(object_stem, recursive=True)
                return
            # Item by item for clarity
            files = cloudi.glob(object_stem)
            for oname in files:
                if verbose:
                    print("> Deleting: %s:%s" % (bucket, oname))
                ob = cloudi.get_object(oname)
                ob.delete()
        else:
            ob = cloudi.get_object(os.path.join(fpath, fname, key))
            ob.delete()


def _named_cloud_cache(fn, *args, **kwargs):
    """Caching of outputs of simple functions in S3 database

    Unclear if this is done right. As written, this adds two sneaky keyword arguments
    to the function call `fn` (k)
    for fn.  I think I want to have this function add
    `sname` and `is_overwrite` keyword arguments to the function it decorates, 
    but maybe not. 
    """

    @functools.wraps(fn)
    def cache_fn(*args, **kwargs):
        cpath = "cloud:mark:cache"
        if "sname" in kwargs:
            sname = kwargs.pop("sname")
        else:
            sname = None
        if "is_overwrite" in kwargs:
            is_overwrite = kwargs.pop("is_overwrite")
        else:
            is_overwrite = False
        if "is_verbose" in kwargs:
            is_verbose = kwargs.pop("is_verbose")
        else:
            is_verbose = False
        if sname is not None:
            fpath = os.path.join(cpath, sname)
            if fexists(fpath) and not is_overwrite:
                if is_verbose:
                    print("Downloading %s..." % sname)
                oo = load_array(fpath, "data")
                return oo
            else:
                # print("=== Inside wrapper, running function... ===")
                out = fn(*args, **kwargs)
                # print("=== Finished with function, saving some shit in %s==="%os.path.join(cpath, sname))
                if is_verbose:
                    print("Storing %s..." % sname)
                save_arrays(fpath, data=out)
            return out
        else:
            raise Exception("sname variable not found!")  # TEMP FOR DEBUGGING
            return fn(*args, **kwargs)

    return cache_fn


def _get_kwargs(fn):
    """Get keyword arguments w/ default values for a function as a dict

    If no keyword args exist for `fn` input, returns an empty dict.
    """
    assert callable(fn), "{} is not a function!".format(fn)
    try:
        kws = inspect.getargspec(fn)
    except TypeError:
        return {}
    if kws.defaults is None:
        return {}
    defaults = dict(zip(kws.args[-len(kws.defaults) :], kws.defaults))
    return defaults


def _cloud_cache(fn, *args, **kwargs):
    """Caching of outputs of simple functions in S3 database (or elsewhere)
    
    Example usage:
    @_cloud_cache
    def my_fun(a, b, c=3):
        return a * b, c**2
    """

    @functools.wraps(fn)
    def cache_fn(*args, **kwargs):
        # Defaults
        is_verbose = True
        cpath = options.config.get("db_dirs", "cache_dir")
        # Hash all inputs to save string
        kws = _get_kwargs(fn)
        kws.update(kwargs)
        inputs = (
            [str(x) for x in args] + list(kws.keys()) + [str(x) for x in kws.values()]
        )
        inputs = str(hash("".join(inputs)))
        sname = fn.__name__ + "_" + inputs.replace("-", "n")
        if fexists(os.path.join(cpath, sname)):
            if is_verbose:
                print("Downloading %s..." % sname)
            output_vars = [
                os.path.split(o)[1] for o in lsfiles(os.path.join(cpath, sname))
            ]
            oo = [load_array(os.path.join(cpath, sname), key) for key in output_vars]
            return oo
        else:
            # print("=== Inside wrapper, running function... ===")
            out = fn(*args, **kwargs)
            # print("=== Finished with function, saving some shit in %s==="%os.path.join(cpath, sname))
            if is_verbose:
                print("Storing %s..." % sname)
            out_dict = dict(("array_%02d" % i, o) for i, o in enumerate(out, 1))
            save_arrays(os.path.join(cpath, sname), **out_dict)
        return out

    return cache_fn


#######################################
### --- Image loading functions --- ###
#######################################


def pil_loader(fpath):
    pil_im = Image.open(path)
    # If alpha channel exists, get rid of it
    bands = pil_im.getbands()  # Returns, e.g., ['R', 'G', 'B', 'A']
    if "A" in bands:
        # Add (white) background
        bg = Image.fromarray(np.ones(pil_im.size + (len(bands),), dtype=np.uint8) * 255)
        pil_im_alpha = Image.alpha_composite(bg, pil_im)
        return pil_im_alpha.convert("RGB")
    else:
        return pil_im.convert("RGB")


if torch_available:
    # Transforms for data input
    def get_xfm(scale=224, center_crop=None, tensor=True, normalize=True, **kwargs):
        """Get pytorch transform for input images
        
        kwargs are meant to be optional inserts into transform sequence, inserted one by one
        into the list (indices in list are given by dict keys) Not working yet.
        """
        xfmlist = []
        if (scale is not None) and (scale is not False):
            # xfmlist.append(transforms.Scale(scale))
            xfmlist.append(transforms.Resize(scale))
        if (center_crop is not None) and (center_crop is not False):
            xfmlist.append(transforms.CenterCrop(center_crop))
        if (tensor is not None) and (tensor is not False):
            xfmlist.append(transforms.ToTensor())
        if (normalize is not None) and (normalize is not False):
            if normalize is True:
                # Default values from ImageNet
                xfmlist.append(
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                )
            else:
                # User-specified normalization values
                xfmlist.append(transforms.Normalize(*normalize))
        xfm = transforms.Compose(xfmlist)
        return xfm

    default_xfm = get_xfm()

    class ImageList(Dataset):
        """Class to load images with no classes / labels, for simple feature extraction"""

        def __init__(
            self,
            images,
            classes=None,
            transform=None,
            target_transform=None,
            loader=pil_loader,
        ):
            """Class to load images

            Parameters
            ----------
            images : list
                List of image file names to load
            classes : list | array
                
            transform : torch transform
                Set of operations to perform on data as it is loaded
            """
            if transform is None:
                transform = default_xfm
            if classes is None:
                classes = np.zeros((len(images),), dtype=np.int)
            self.imgs = list(zip(images, classes))
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        def __len__(self):
            return len(self.imgs)

    class ImageArray(Dataset):
        """Class to load images with no classes / labels, for simple feature extraction"""

        def __init__(
            self,
            images,
            classes=None,
            transform=None,
            target_transform=None,
            loader=pil_loader,
        ):
            """Class to load images

            Parameters
            ----------
            images : array-like (possibly open hdf file)
                size is [n, h, w, rgb] 
                or [n, h, w]
            classes : list | array
                labels for each image (`n` long array or list)
            transform : torch transform
                Set of operations to perform on data as it is loaded
            """
            self.imgs = images  # np.rollaxis(images, -1, 0)
            if transform is None:
                transform = default_xfm
            if classes is None:
                classes = np.zeros((len(self.imgs),), dtype=np.int)
            self.classes = classes
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):

            img = self.imgs[index]
            if np.ndim(img) == 2:
                img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
            img = Image.fromarray((img * 255).astype(np.uint8))
            target = self.classes[index]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        def __len__(self):
            return len(self.imgs)

    class SimpleImageFolder(Dataset):
        """Class to load images with no classes / labels, for simple feature extraction"""

        def __init__(
            self,
            images,
            transform=default_xfm,
            target_transform=None,
            loader=pil_loader,
        ):
            """Class to load images

            Parameters
            ----------
            images : list
                List of image file names to load
            transform : torch transform
                Set of operations to perform on data as it is loaded
                see module transforms.py
            """
            self.imgs = zip(images, np.zeros((len(images),), dtype=np.int))
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        def __len__(self):
            return len(self.imgs)

    class ImageFolder(Dataset):
        """Load images based on text file of image file names + classification categories"""

        def __init__(
            self,
            list_file,
            transform=default_xfm,
            target_transform=None,
            loader=pil_loader,
            class_to_idx=None,
        ):
            """Load images based on text file of image file names + classification categories

            Parameters
            ----------
            list_file : string
                file name for list of images to load. File should be formatted as:
                image_file.ext <space> target_class
                (one row per image to be loaded)
            transform : pytorch transform series
                series of transforms (clipping, rotation, normalization, mapping to tensor, etc)
                to be applied to images at load time.
            target_tranform : transformation of target
                Map target to potential other target class
            loader : pytorch loader
                ...
            class_to_idx : dict
                dictionary to map classes to class numbers (classification target indices)
                if not provided, attempts to read from file structure; if it fails, set to None
            """
            images = []
            lines = file(list_file).read().split("\n")
            do_class_to_idx = class_to_idx is None
            if do_class_to_idx:
                class_to_idx = {}
            imgs = []
            for l in lines:
                if not l.strip():
                    continue
                fname, label_num = l.split(" ")
                label_num = int(label_num)
                if do_class_to_idx:
                    try:
                        fnComps = fname.split("/")
                        class_name = "_".join(fnComps[-2].split("_")[:-1])
                        class_to_idx[class_name] = label_num
                    except:
                        # No class_to_idx dict definable
                        pass
                imgs.append((fname, label_num))
            try:
                classes = class_to_idx.keys()
                classes.sort(key=lambda x: class_to_idx[x])
            except:
                print("Failed to find class names")
                classes = []
            self.imgs = imgs
            self.classes = classes
            self.class_to_idx = class_to_idx
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        def __len__(self):
            return len(self.imgs)

    class HDFDataSet(Dataset):
        """Class to load images from hdf files"""

        def __init__(self, fname, variable_name="images", ims_per_file=None):
            self.datasets = []
            self.total_count = 0
            for i, f in enumerate(hdf5_list):
                with h5py.File(f, "r") as hf:
                    dataset = hf[variable_name].value
                self.datasets.append(dataset)
                self.total_count += len(dataset)

        def __getitem__(self, index):
            """
            Suppose each hdf5 file has 10000 samples
            """
            dataset_index = index % 10000
            in_dataset_index = int(index / 10000)
            return self.datasets[dataset_index][in_dataset_index]

        def __len__(self):
            return len(self.total_count)


if opencv_available:

    def load_exr_normals(
        fname, xflip=True, yflip=True, zflip=True, clip=True, zero_norm_to_nan=False
    ):
        """Load an exr (floating point) image to surface normal array

        """
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        imc = img - 1
        y, z, x = imc.T
        if xflip:
            x = -x
        if yflip:
            y = -y
        if zflip:
            z = -z
        imc = np.dstack([x.T, y.T, z.T])
        if clip:
            imc = np.clip(imc, -1, 1)
        if zero_norm_to_nan:
            pass
        return imc

    def load_exr_zdepth(fname, thresh=1000):
        """Load an exr (floating point) image to absolute distance array"""
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        z = img[..., 0]
        z[z > thresh] = np.nan
        return z


def save_movie(
    fname,
    array,
    fps=30,
    crf=0,
    preset="fast",
    codec="libx264",
    color_format="rgb24",
    is_verbose=False,
):
    """Save array of images as an mp4 movie"""
    ff = VideoEncoderFFMPEG(
        fname,
        array.shape[1:3],
        fps=fps,
        color_format=color_format,
        codec=codec,
        preset=preset,
        crf=crf,
        is_verbose=is_verbose,
    )
    ff.write(array)
    ff.stop()


class VideoEncoderFFMPEG(object):
    """ Base class for encoder interfaces. """

    def __init__(
        self,
        fname,
        resolution,
        fps,
        color_format="rgb24",
        codec="libx264",
        preset="fast",
        crf=0,
        is_verbose=False,
    ):
        """ Constructor.

        Parameters
        ----------
        fname: str
            File name for movie to be written.
        resolution: tuple, len 2
            Desired (horizontal, vertical) resolution.
        fps: int
            Desired refresh rate.
        color_format: str, default 'rgb24'
            The target color format. Set to 'gray' grayscale
        codec: str, default 'libx264'
            The desired video codec.
        """
        self.fname = fname
        if os.path.exists(self.fname):
            os.remove(self.fname)
        self.resolution = resolution
        self.fps = fps
        self.color_format = color_format
        self.codec = codec
        self.preset = preset
        self.crf = crf
        self.is_verbose = is_verbose
        # Business
        ffmpeg_cmd = self._get_ffmpeg_cmd()
        if is_verbose:
            print("FFMPEG_cmd:", ffmpeg_cmd)
        self.video_writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    def _get_ffmpeg_cmd(self):
        """ Get the FFMPEG command to start the sub-process. """
        size = "{}x{}".format(self.resolution[1], self.resolution[0])
        print("size: ", size)
        if self.preset is None:
            return [
                "ffmpeg",
                # -- Input -- #
                "-an",  # no audio
                "-r",
                str(self.fps),  # fps
                "-f",
                "rawvideo",  # format
                "-s",
                size,  # resolution
                "-pix_fmt",
                self.color_format,  # color format
                "-i",
                "pipe:",  # piped to stdin
                # -- Output -- #
                "-c:v",
                codec,  # video codec
                self.fname,
            ]
        else:
            return [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                # -- Input -- #
                "-an",  # no audio
                "-r",
                str(self.fps),  # fps
                "-f",
                "rawvideo",  # format
                "-s",
                size,  # resolution
                "-pix_fmt",
                self.color_format,  # color format
                "-i",
                "pipe:",  # piped to stdin
                "-preset",
                self.preset,
                "-crf",
                str(self.crf),
                # -- Output -- #
                "-c:v",
                self.codec,  # video codec
                self.fname,
            ]

    def write(self, img):
        """ Write a frame to disk.

        Parameters
        ----------
        img : array_like
            The input frame or frames. To write multiple frames, array should be 
            [time, y, x, color]
        Notes
        -----
        Converts input images to uint8 - this could involve some loss of precision!
        """
        if np.ndim(img) == 4:
            for img_ in tqdm(img):
                self.write(img_)
            return
        if img.dtype in (np.uint8,):
            to_write = img
        else:
            if img.max() > 1:
                # NOTE: if file has max of 100 (as in LAB converted
                # grayscale images), this may cause aliasing (loss of precision)
                to_write = img.astype(np.uint8)
            else:
                to_write = (img * 255).astype(np.uint8)
        self.video_writer.stdin.write(to_write.tostring())

    def stop(self):
        self.video_writer.stdin.close()


def write_ffmpeg_images(
    file_pattern, output_file, fps=30, size=None, codec="libx264", preset="fast", crf=0,
):
    """Still failing. Trying to call e.g.:
    ffmpeg -f image2 -pattern_type glob -i '/path/to/dir/*png' -r 30 -s 500x500 -c:v libx264 -preset fast -an -crf 0 /path/to/output/my_movie.mp4
    """
    cmd = [
        "ffmpeg",
        "-f",
        "image2",
        "-pattern_type",
        "glob",
        "-i",
        "'%s'" % file_pattern,
        "-r",
        "%d" % fps,
        "-s",
        "{}x{}".format(size[0], size[1]),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-an",
        "-crf",
        "%d" % crf,
        output_file,
    ]
    out, err = subprocess.check_output(cmd)


# Stubs. Good ideas, from https://discuss.pytorch.org/t/use-of-dataset-class/1620/4
# class MergedDataset(Dataset):
#     """Class to load images from hdf files"""
#     def __init__(self, hdf5_list, ims_per_file=None):
#         self.datasets = []
#         self.total_count = 0
#         for i, f in enumerate(hdf5_list):
#            h5_file = h5py.File(f, 'r')
#            dataset = h5_file['YOUR DATASET NAME']
#            self.datasets.append(dataset)
#            self.total_count += len(dataset)

#     def __getitem__(self, index):
#         '''
#         Suppose each hdf5 file has 10000 samples
#         '''
#         dataset_index = index % 10000
#         in_dataset_index = int(index / 10000)
#         return self.datasets[dataset_index][in_dataset_index]

#     def __len__(self):
#         return len(self.total_count)

# class CloudHDFDataSet(Dataset):
#   def __init__(self, cloud_paths):

#       hdf5_list = [x for x in glob.glob(os.path.join(path_patients,'*.h5'))]#only h5 files
#       print 'h5 list ',hdf5_list
#       self.datasets = []
#       self.datasets_gt=[]
#       self.total_count = 0
#       self.limits=[]
#       for f in hdf5_list:
#          h5_file = h5py.File(f, 'r')
#          dataset = h5_file['data']
#          dataset_gt = h5_file['label']
#          self.datasets.append(dataset)
#          self.datasets_gt.append(dataset_gt)
#          self.limits.append(self.total_count)
#          self.total_count += len(dataset)
#          #print 'len ',len(dataset)
#       #print self.limits

#   def __getitem__(self, index):

#       dataset_index=-1
#       #print 'index ',index
#       for i in xrange(len(self.limits)-1,-1,-1):
#         #print 'i ',i
#         if index>=self.limits[i]:
#           dataset_index=i
#           break
#       #print 'dataset_index ',dataset_index
#       assert dataset_index>=0, 'negative chunk'

#       in_dataset_index = index-self.limits[dataset_index]

#       return self.datasets[dataset_index][in_dataset_index], self.datasets_gt[dataset_index][in_dataset_index]

#   def __len__(self):
#       return self.total_count
