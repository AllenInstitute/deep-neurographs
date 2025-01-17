"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Helper routines for processing images.

"""

from abc import ABC, abstractmethod
from skimage.color import label2rgb

import numpy as np
import s3fs
import tensorstore as ts
import zarr

from deep_neurographs.utils import util


class ImageReader(ABC):
    """
    Abstract class to create image readers classes.

    """

    def __init__(self, img_path):
        """
        Class constructor of image reader.

        Parameters
        ----------
        img_path : str
            Path to image.

        Returns
        -------
        None

        """
        self.img = None
        self.img_path = img_path
        self._load_image()

    @abstractmethod
    def _load_image(self):
        """
        This method should be implemented by subclasses to load the image
        based on img_path.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        pass

    def read(self, voxel, shape, from_center=True):
        """
        Reads a patch from an image given a voxel coordinate and patch shape.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel coordinate that is either the center or top-left-front
            corner of the image patch to be read.
        shape : Tuple[int]
            Shape of the image patch to be read.
        from_center : bool, optional
            Indication of whether "voxel" is the center or top-left-front
            corner of the image patch to be read. The default is True.

        Returns
        -------
        ArrayLike
            Image patch.

        """
        s, e = get_start_end(voxel, shape, from_center=from_center)
        if len(self.shape()) == 3:
            return self.img[s[0]: e[0], s[1]: e[1], s[2]: e[2]]
        elif len(self.shape()) == 5:
            return self.img[0, 0, s[0]: e[0], s[1]: e[1], s[2]: e[2]]
        else:
            raise ValueError(f"Unsupported image shape: {self.shape()}")

    def read_with_bbox(self, bbox):
        """
        Reads an image patch by using a "bbox".

        Parameters
        ----------
        bbox : dict
            Dictionary that contains min and max coordinates of a bounding
            box.

        Returns
        -------
        numpy.ndarray
            Image patch.

        """
        try:
            shape = [bbox["max"][i] - bbox["min"][i] for i in range(3)]
            return self.read(bbox["min"], shape, from_center=False)
        except Exception:
            return np.zeros(shape)

    def read_profile(self, spec):
        """
        Reads an intensity profile from an image (i.e. image profile).

        Parameters
        ----------
        spec : dict
            Dictionary that stores the bounding box of patch to be read and the
            voxel coordinates of the profile path.

        Returns
        -------
        List[float]
            Image profile.

        """
        img_patch = normalize(self.read_with_bbox(spec["bbox"]))
        return [img_patch[voxel] for voxel in map(tuple, spec["profile_path"])]

    def shape(self):
        """
        Gets the shape of image.

        Parameters
        ----------
        None

        Returns
        -------
        Tuple[int]
            Shape of image.

        """
        return self.img.shape


class TensorStoreReader(ImageReader):
    """
    Class that reads image with tensorstore.

    """

    def __init__(self, img_path, driver):
        """
        Constructs a TensorStore image reader.

        Parameters
        ----------
        img_path : str
            Path to image.
        driver : str
            Storage driver needed to read the image at "path".

        Returns
        -------
        None

        """
        self.driver = driver
        super().__init__(img_path)

    def _load_image(self):
        """
        Load the image using the TensorStore library.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.img = ts.open(
            {
                "driver": self.driver,
                "kvstore": {
                    "driver": "gcs",
                    "bucket": "allen-nd-goog",
                    "path": self.img_path,
                },
                "context": {
                    "cache_pool": {"total_bytes_limit": 1000000000},
                    "cache_pool#remote": {"total_bytes_limit": 1000000000},
                    "data_copy_concurrency": {"limit": 8},
                },
                "recheck_cached_data": "open",
            }
        ).result()
        if self.driver == "neuroglancer_precomputed":
            self.img = self.img[ts.d["channel"][0]]
            self.img = self.img[ts.d[0].transpose[2]]
            self.img = self.img[ts.d[0].transpose[1]]

    def read(self, voxel, shape, from_center=True):
        img_patch = super().read(voxel, shape, from_center)
        return img_patch.read().result()


class ZarrReader(ImageReader):
    """
    Class that reads image with zarr.

    """

    def __init__(self, img_path):
        """
        Constructs a TensorStore image reader.

        Parameters
        ----------
        img_path : str
            Path to image.

        Returns
        -------
        None

        """
        super().__init__(img_path)

    def _load_image(self):
        """
        Load the image using the zarr library.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        store = s3fs.S3Map(root=self.img_path, s3=s3fs.S3FileSystem(anon=True))
        self.img = zarr.open(store, mode="r")


def get_start_end(voxel, shape, from_center=True):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    voxel : tuple
        Voxel coordinate that is either the center or top-left-front corner of
        the image patch to be read.
    shape : Tuple[int]
        Shape of the image patch to be read.
    from_center : bool, optional
        Indication of whether "voxel" is the center or top-left-front corner
        of the image patch to be read. The default is True.

    Return
    ------
    Tuple[List[int]]
        Start and end indices of the image patch to be read.

    """
    if from_center:
        start = [voxel[i] - shape[i] // 2 for i in range(3)]
        end = [voxel[i] + shape[i] // 2 for i in range(3)]
    else:
        start = voxel
        end = [voxel[i] + shape[i] for i in range(3)]
    return start, end


# -- Operations --
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
    "axis". This routine differs from "get_mip" because it retuns an RGB
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


def get_profile(img_reader, spec, profile_id):
    """
    Gets the image profile for a given proposal.

    Parameters
    ----------
    img_reader : ImageReader
        Image reader.
    spec : dict
        Dictionary that contains the image bounding box and coordinates of the
        image profile path.
    profile_id : Frozenset[int]
        Identifier of profile.

    Returns
    -------
    dict
        Dictionary that maps an id (e.g. node, edge, or proposal) to its image
        profile.

    """
    profile = img_reader.read_profile(spec)
    avg, std = util.get_avg_std(profile)
    profile.extend([avg, std])
    return {profile_id: profile}


# --- Coordinate Conversions ---
def to_physical(voxel, anisotropy, shift=[0, 0, 0]):
    """
    Converts a voxel coordinate to a physical coordinate by applying the
    anisotropy scaling factors.

    Parameters
    ----------
    coord : ArrayLike
        Coordinate to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    shift : ArrayLike, optional
        Shift to be applied to "coord". The default is [0, 0, 0].

    Returns
    -------
    Tuple[float]
        Converted coordinates.

    """
    voxel = voxel[::-1]
    return tuple([voxel[i] * anisotropy[i] - shift[i] for i in range(3)])


def to_voxels(xyz, anisotropy, multiscale=0):
    """
    Converts coordinate from a physical to voxel space.

    Parameters
    ----------
    xyz : ArrayLike
        Physical coordiante to be converted to a voxel coordinate.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
        The default is 0.

    Returns
    -------
    numpy.ndarray
        Voxel coordinate.

    """
    scaling_factor = 1.0 / 2 ** multiscale
    voxel = [scaling_factor * xyz[i] / anisotropy[i] for i in range(3)]
    return np.round(voxel[::-1]).astype(int)


# -- utils --
def init_bbox(origin, shape):
    """
    Gets the min and max coordinates of a bounding box based on "origin" and
    "shape".

    Parameters
    ----------
    origin : tuple[int]
        Voxel coordinate of the origin of the bounding box, which is assumed
        to be top-front-left corner.
    shape : Tuple[int]
        Shape of the bounding box.

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
