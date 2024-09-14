"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from copy import deepcopy

import fastremap
import numpy as np
import tensorstore as ts
from skimage.color import label2rgb

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


def read_tensorstore_with_bbox(img, bbox):
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
    except Exception as e:
        print(type(e), e)
        return np.zeros(shape)


def read_profile(img, specs):
    """
    Reads an intensity profile from an image (i.e. image profile).

    Parameters
    ----------
    img : tensorstore.TensorStore
        Image to be read.
    specs : dict
        Dictionary that stores the bounding box of chunk to be read and the
        voxel coordinates of the profile path.

    Returns
    -------
    numpy.ndarray
        Image profile.

    """
    img_chunk = normalize(read_tensorstore_with_bbox(img, specs["bbox"]))
    return read_intensities(img_chunk, specs["profile_path"])


def read_intensities(img, voxels):
    """
    Reads the image intensities of voxels.

    Parameters
    ----------
    img : tensorstore.TensorStore
        Image to be read.
    voxels : list
        Voxels to be read.

    Returns
    -------
    list
        Image intensities.

    """
    return [img[tuple(voxel)] for voxel in voxels]


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
        end = [voxel[i] + shape[i] + 1 for i in range(3)]
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


# --- coordinate conversions ---
def img_to_patch(voxel, patch_centroid, patch_shape):
    """
    Converts coordinates from global to local image coordinates.

    Parameters
    ----------
    voxel : numpy.ndarray
        Voxel coordinate to be converted.
    patch_centroid : numpy.ndarray
        Centroid of image patch.
    patch_shape : numpy.ndarray
        Shape of image patch.

    Returns
    -------
    tuple
        Converted coordinates.

    """
    half_patch_shape = [patch_shape[i] // 2 for i in range(3)]
    patch_voxel = voxel - patch_centroid + half_patch_shape
    return tuple(patch_voxel.astype(int))


def patch_to_img(voxel, patch_centroid, patch_dims):
    """
    Converts coordinates from local to global image coordinates.

    Parameters
    ----------
    coord : numpy.ndarray
        Coordinates to be converted.
    patch_centroid : numpy.ndarray
        Centroid of image patch.
    patch_shape : numpy.ndarray
        Shape of image patch.

    Returns
    -------
    tuple
        Converted coordinates.

    """
    half_patch_dims = [patch_dims[i] // 2 for i in range(3)]
    return np.round(voxel + patch_centroid - half_patch_dims).astype(int)


def to_world(voxel, anisotropy=ANISOTROPY, shift=[0, 0, 0]):
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
    return tuple([voxel[i] * anisotropy[i] - shift[i] for i in range(3)])


def to_voxels(xyz, anisotropy=ANISOTROPY, downsample_factor=0):
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
    voxel = downsample_factor * (xyz / np.array(anisotropy))
    return np.round(voxel).astype(int)


# -- utils --
def get_bbox(origin, shape):
    """
    Gets the min and max coordinates of a bounding box based on "origin" and
    "shape".

    Parameters
    ----------
    origin : tuple
        Origin of bounding box which is assumed to be top, front, left corner.
    shape : tuple
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
        "min": np.floor(np.min(voxels, axis=0) - 1).astype(int),
        "max": np.ceil(np.max(voxels, axis=0) + buffer + 1).astype(int),
    }
    return bbox


def get_chunk_labels(path, xyz, shape, from_center=True):
    """
    Gets the labels of segments contained in chunk centered at "xyz".

    Parameters
    ----------
    path : str
        Path to segmentation stored in a GCS bucket.
    xyz : numpy.ndarray
        Center point of chunk to be read.
    shape : tuple
        Shape of chunk to be read.
    from_center : bool, optional
        Indication of whether "xyz" is the center point or upper, left, front
        corner of chunk to be read. The default is True.

    Returns
    -------
    set
        Labels of segments contained in chunk read from GCS bucket.

    """
    img = open_tensorstore(path)
    img = read_tensorstore(img, xyz, shape, from_center=from_center)
    return set(fastremap.unique(img).astype(int))
