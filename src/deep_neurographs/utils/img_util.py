"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from copy import deepcopy

import numpy as np
import tensorstore as ts
from skimage.color import label2rgb
from tifffile import imwrite

from deep_neurographs.utils import util

ANISOTROPY = [0.748, 0.748, 1.0]
SUPPORTED_DRIVERS = ["neuroglancer_precomputed", "n5", "zarr"]


# --- io utils ---
def open_tensorstore(path, driver="neuroglancer_precomputed"):
    """
    Opens an image that is assumed to be stored as a directory of shard files.

    Parameters
    ----------
    path : str
        Path to directory containing shard files.
    driver : str, optional
        Storage driver needed to read data at "path". The default is
        "neuroglancer_precomputed".

    Returns
    -------
    tensorstore.TensorStore
        Pointer to image stored at "path" in a GCS bucket.

    """
    assert driver in SUPPORTED_DRIVERS, "Driver is not supported!"
    img = ts.open(
        {
            "driver": driver,
            "kvstore": {
                "driver": "gcs",
                "bucket": "allen-nd-goog",
                "path": path,
            },
            "context": {
                "cache_pool": {"total_bytes_limit": 1000000000},
                "cache_pool#remote": {"total_bytes_limit": 1000000000},
                "data_copy_concurrency": {"limit": 8},
            },
            "recheck_cached_data": "open",
        }
    ).result()
    if driver == "neuroglancer_precomputed":
        return img[ts.d["channel"][0]]
    elif driver == "zarr":
        img = img[0, 0, :, :, :]
        img = img[ts.d[0].transpose[2]]
        img = img[ts.d[0].transpose[1]]
    return img


def read(img, voxel, shape, from_center=True):
    """
    Reads a chunk of data from an image given a voxel coordinate and shape.

    Parameters
    ----------
    img : numpy.ndarray
        Image to be read.
    voxel : tuple
        Voxel coordinate that specifies either the center or top, left, front
        corner of the chunk to be read.
    shape : tuple
        Shape (dimensions) of the chunk to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the chunk or the top, left, front corner. The default is True.

    Returns
    -------
    numpy.ndarray
        Chunk of data read from an image.

    """
    start, end = get_start_end(voxel, shape, from_center=from_center)
    return deepcopy(
        img[start[0]: end[0], start[1]: end[1], start[2]: end[2]]
    )


def read_tensorstore(img, voxel, shape, from_center=True):
    """
    Reads a chunk from an image given a voxel coordinate and the desired shape
    of the chunk.

    Parameters
    ----------
    img : tensorstore.TensorStore
        Image to be read.
    voxel : tuple
        Voxel coordinate that specifies either the center or top, left, front
        corner of the chunk to be read.
    shape : tuple
        Shape (dimensions) of the chunk to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the chunk or the starting point. The default is True.

    Returns
    -------
    numpy.ndarray
        Chunk of data read from an image.

    """
    return read(img, voxel, shape, from_center=from_center).read().result()


def read_tensorstore_with_bbox(img, bbox, normalize=True):
    """
    Reads a chunk from a subarray that is determined by "bbox".

    Parameters
    ----------
    img : tensorstore.TensorStore
        Image to be read.
    bbox : dict
        Dictionary that contains min and max coordinates of a bounding box.

    Returns
    -------
    numpy.ndarray
        Chunk of data read from an image.

    """
    try:
        shape = [bbox["max"][i] - bbox["min"][i] for i in range(3)]
        return read_tensorstore(img, bbox["min"], shape, from_center=False)
    except Exception:
        print(f"Unable to read from image with bbox {bbox}")
        return np.zeros(shape)


def read_profile(img, spec):
    """
    Reads an intensity profile from an image (i.e. image profile).

    Parameters
    ----------
    img : tensorstore.TensorStore
        Image to be read.
    spec : dict
        Dictionary that stores the bounding box of chunk to be read and the
        voxel coordinates of the profile path.

    Returns
    -------
    numpy.ndarray
        Image profile.

    """
    img_patch = normalize(read_tensorstore_with_bbox(img, spec["bbox"]))
    return [img_patch[voxel] for voxel in map(tuple, spec["profile_path"])]


def get_start_end(voxel, shape, from_center=True):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    voxel : tuple
        Voxel coordinate that specifies either the center or top, left, front
        corner of the chunk to be read.
    shape : tuple
        Shape (dimensions) of the chunk to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the chunk or the starting point. The default is True.

    Return
    ------
    tuple[list[int]]
        Start and end indices of the chunk to be read.

    """
    if from_center:
        start = [voxel[i] - shape[i] // 2 for i in range(3)]
        end = [voxel[i] + shape[i] // 2 for i in range(3)]
    else:
        start = voxel
        end = [voxel[i] + shape[i] for i in range(3)]
    return start, end


# -- operations --
def normalize(img):
    """
    Normalizes an image so that the minimum and maximum intensity values are 0
    and 1.

    Parameters
    ----------
    img : numpy.ndarray
        Image to be normalized.

    Returns
    -------
    numpy.ndarray
        Normalized image.

    """
    img -= np.min(img)
    return img / max(1, np.max(img))


def get_mip(img, axis=0):
    """
    Compute the maximum intensity projection (MIP) of "img" along "axis".

    Parameters
    ----------
    img : numpy.ndarray
        Image to compute MIP of.
    axis : int, optional
        Projection axis. The default is 0.

    Returns
    -------
    numpy.ndarray
        MIP of "img".

    """
    return np.max(img, axis=axis)


def get_labels_mip(img, axis=0):
    """
    Compute the maximum intensity projection (MIP) of a segmentation along
    "axis". This routine differs from "get_mip" because it retuns an rgb
    image.

    Parameters
    ----------
    img : numpy.ndarray
        Image to compute MIP of.
    axis : int, optional
        Projection axis. The default is 0.

    Returns
    -------
    numpy.ndarray
        MIP of "img".

    """
    mip = np.max(img, axis=axis)
    mip = label2rgb(mip)
    return (255 * mip).astype(np.uint8)


def get_profile(img, spec, profile_id):
    """
    Gets the image profile for a given proposal.

    Parameters
    ----------
    img : tensorstore.TensorStore
        Image that profiles are generated from.
    spec : dict
        Dictionary that contains the image bounding box and coordinates of the
        image profile path.
    profile_id : frozenset
        Identifier of profile.

    Returns
    -------
    dict
        Dictionary that maps an id (e.g. node, edge, or proposal) to its image
        profile.

    """
    profile = read_profile(img, spec)
    avg, std = util.get_avg_std(profile)
    profile.extend([avg, std])
    return {profile_id: profile}


# --- coordinate conversions ---
def to_world(voxel, shift=[0, 0, 0]):
    """
    Converts coordinates from voxels to world.

    Parameters
    ----------
    coord : numpy.ndarray
        Coordinate to be converted.
    shift : list, optional
        Shift to be applied to "coord". The default is [0, 0, 0].

    Returns
    -------
    tuple
        Converted coordinates.

    """
    return tuple([voxel[i] * ANISOTROPY[i] - shift[i] for i in range(3)])


def to_voxels(xyz, downsample_factor=0):
    """
    Converts coordinates from world to voxel.

    Parameters
    ----------
    xyz : numpy.ndarray
        xyz coordinate to be converted to voxels.
    anisotropy : list, optional
        Anisotropy to be applied to values of interest. The default is
        [1.0, 1.0, 1.0].
    downsample_factor : int, optional
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into. The default is 0.

    Returns
    -------
    numpy.ndarray
        Coordinates converted to voxels.

    """
    downsample_factor = 1.0 / 2 ** downsample_factor
    voxel = downsample_factor * (xyz / np.array(ANISOTROPY))
    return np.round(voxel).astype(int)


# -- utils --
def init_bbox(origin, shape):
    """
    Gets the min and max coordinates of a bounding box based on "origin" and
    "shape".

    Parameters
    ----------
    origin : tuple[int]
        Origin of bounding box which is assumed to be top, front, left corner.
    shape : tuple[int]
        Shape of bounding box.

    Returns
    -------
    dict or None
        Bounding box.

    """
    if origin and shape:
        origin = np.array(origin)
        shape = np.array(shape)
        return {"min": origin, "max": origin + shape}
    else:
        return None


def get_minimal_bbox(voxels, buffer=0):
    """
    Gets the min and max coordinates of a bounding box that contains "voxels".

    Parameters
    ----------
    voxels : numpy.ndarray
        Array containing voxel coordinates.
    buffer : int, optional
        Constant value added/subtracted from the max/min coordinates of the
        bounding box. The default is 0.

    Returns
    -------
    dict
        Bounding box.

    """
    bbox = {
        "min": np.floor(np.min(voxels, axis=0) - buffer).astype(int),
        "max": np.ceil(np.max(voxels, axis=0) + buffer).astype(int),
    }
    return bbox


def find_img_path(bucket_name, img_root, dataset_name):
    """
    Find the path of a specific dataset in a GCS bucket.

    Parameters:
    ----------
    bucket_name : str
        Name of the GCS bucket where the images are stored.
    img_root : str
        Root directory path in the GCS bucket where the images are located.
    dataset_name : str
        Name of the dataset to be searched for within the subdirectories.

    Returns:
    -------
    str
        Path of the found dataset subdirectory within the specified GCS bucket.

    """
    for subdir in util.list_gcs_subdirectories(bucket_name, img_root):
        if dataset_name in subdir:
            return subdir + "whole-brain/fused.zarr/"
    raise f"Dataset not found in {bucket_name} - {img_root}"
