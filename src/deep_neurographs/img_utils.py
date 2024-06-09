"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import tensorstore as ts
import zarr
from skimage.color import label2rgb

SUPPORTED_DRIVERS = ["neuroglancer_precomputed", "n5", "zarr"]


# --- io utils ---
def open_tensorstore(path, driver):
    """
    Uploads segmentation mask stored as a directory of shard files.

    Parameters
    ----------
    path : str
        Path to directory containing shard files.
    driver : str
        Storage driver needed to read data at "path".

    Returns
    -------
    dict
        Sparse image volume.

    """
    assert driver in SUPPORTED_DRIVERS, "Error! Driver is not supported!"
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


def open_zarr(path):
    """
    Opens zarr file at "path".

    Parameters
    ----------
    path : str
        Path to zarr file to be opened.

    Returns
    -------
    np.ndarray
        Contents of zarr file at "path".

    """
    n5store = zarr.N5FSStore(path, "r")
    if "653980" in path:
        return zarr.open(n5store).ch488.s0
    elif "653158" in path:
        return zarr.open(n5store).s0


def read_tensorstore(img, xyz, shape, from_center=True):
    """
    Reads a chunk of data from the specified tensorstore array, given the xyz
    coordinates and shape of the chunk.

    Parameters
    ----------
    img : tensorstore.TensorStore
        Image to be read.
    xyz : tuple
        xyz coordinates of chunk to be read.
    shape : tuple
        Shape (dimensions) of the chunk to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the chunk or the starting point. If True, coordinates are the center;
        if False, coordinates are the starting point of the chunk. The default
        is True.

    Returns
    -------
    numpy.ndarray
        Chunk of data read from the image.

    """
    chunk = read_chunk(img, xyz, shape, from_center=from_center)
    return chunk.read().result()


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
        Chunk of data read from the image.
    """
    start = bbox["min"]
    end = bbox["max"]
    try:
        return (
            img[start[0]: end[0], start[1]: end[1], start[2]: end[2]]
            .read()
            .result()
        )
    except Exception as e:
        print(type(e), e)
        shape = [end[i] - start[i] +1 for i in range(3)]
        return np.zeros(shape)

def read_chunk(img, xyz, shape, from_center=True):
    """
    Reads a chunk of data from arr"", given the xyz coordinates and shape of
    the chunk.

    Parameters
    ----------
    img : numpy.ndarray or tensorstore.TensorStore
        Image to be read.
    xyz : tuple
        xyz coordinates of chunk to be read.
    shape : tuple
        Shape (dimensions) of the chunk to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the chunk or the starting point. If True, coordinates are the center;
        if False, coordinates are the starting point of the chunk. The default
        is True.

    Returns
    -------
    numpy.ndarray
        Chunk of data read from the array.

    """
    start, end = get_start_end(xyz, shape, from_center=from_center)
    return deepcopy(
        img[start[0]: end[0], start[1]: end[1], start[2]: end[2]]
    )


def get_start_end(xyz, shape, from_center=True):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    xyz : tuple
        xyz coordinates of chunk to be read.
    shape : tuple
        Shape (dimensions) of the chunk to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the chunk or the starting point. If True, coordinates are the center;
        if False, coordinates are the starting point of the chunk. The default
        is True.

    Return
    ------
    tuple[list[int]]
        Start and end indices of the chunk to be read.

    """
    if from_center:
        start = [xyz[i] - shape[i] // 2 for i in range(3)]
        end = [xyz[i] + shape[i] // 2 for i in range(3)]
    else:
        start = xyz
        end = [xyz[i] + shape[i] for i in range(3)]
    return start, end


def read_superchunks(img_path, labels_path, xyz, shape, from_center=True):
    """
    Reads an image and label mask via multithreading.

    Parameters
    ----------
    img_path : str
        Path to an image on a GCS bucket.
    labels_path : str
        Path to a label mask on a GCS bucket.
    xyz : tuple
        xyz coordinates of chunk to be read.
    shape : tuple
        Shape (dimensions) of the chunk to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the chunk or the starting point. If True, coordinates are the center;
        if False, coordinates are the starting point of the chunk. The default
        is True.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Image and label mask read from cloud bucket.

    """
    with ThreadPoolExecutor() as executor:
        img_job = executor.submit(
            get_superchunk,
            img_path,
            "n5" if ".n5" in img_path else "zarr",
            xyz,
            shape,
            from_center=from_center,
        )
        labels_job = executor.submit(
            get_superchunk,
            labels_path,
            "neuroglancer_precomputed",
            xyz,
            shape,
            from_center=from_center,
        )
    img = img_job.result().astype(np.int16)
    labels = labels_job.result().astype(np.int64)
    assert img.shape == labels.shape, "img.shape != labels.shape"
    return img, labels


def get_superchunk(path, driver, xyz, shape, from_center=True):
    """
    Opens and reads a chunk from an image on a GCS bucket.

    Parameters
    ----------
    path : str
        Path to image on GCS bucket.
    driver : str
        Driver needed to open image.
    xyz : tuple
        xyz coordinates of chunk to be read.
    shape : tuple
        Shape (dimensions) of the chunk to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the chunk or the starting point. If True, coordinates are the center;
        if False, coordinates are the starting point of the chunk. The default
        is True.

    Returns
    -------
    numpy.ndarray
        Image read from GCS bucket.

    """
    img = open_tensorstore(path, driver)
    return read_tensorstore(img, xyz, shape, from_center=from_center)


# -- Image Operations --
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
