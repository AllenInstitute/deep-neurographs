"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Helper routines for reading and processing images.

"""

from abc import ABC, abstractmethod
from scipy.ndimage import zoom
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
            corner of the image patch to be read. Default is True.

        Returns
        -------
        numpy.ndarray
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
        Reads the image patch defined by a given bounding box.

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
        shape = [bbox["max"][i] - bbox["min"][i] for i in range(3)]
        try:
            return self.read(bbox["min"], shape, from_center=False)
        except Exception:
            return np.ones(shape)

    def read_voxel(self, voxel, thread_id=None):
        """
        Reads the intensity value at a given voxel.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel to be read.
        thread_id : Any
            Identifier associated with output. Default is None.

        Returns
        -------
        int
            Intensity value at voxel.
        """
        return thread_id, self.img[voxel]

    def read_profile(self, spec):
        """
        Reads an intensity profile from an image (i.e. image profile).

        Parameters
        ----------
        spec : dict
            Dictionary with the bounding box of the image patch to be read and
            voxel coordinates of the profile path.

        Returns
        -------
        List[float]
            Image profile.
        """
        img_patch = normalize(self.read_with_bbox(spec["bbox"]))
        return [img_patch[v] for v in map(tuple, spec["profile_path"])]

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
    Class that reads an image with TensorStore library.

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
        self.driver = self.init_driver(img_path)
        super().__init__(img_path)

    def init_driver(self, img_path):
        """
        Gets the storage driver needed to read the image.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Storage driver needed to read the image.
        """
        if ".zarr" in img_path:
            return "zarr"
        elif ".n5" in img_path:
            return "n5"
        else:
            return "neuroglancer_precomputed"

    def _load_image(self):
        """
        Loads image using the TensorStore library.

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
        elif self.driver == "n5":
            self.img = self.img[ts.d[0].transpose[2]]
            self.img = self.img[ts.d[0].transpose[1]]

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
            corner of the image patch to be read. Default is True.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        try:
            img_patch = super().read(voxel, shape, from_center)
            return img_patch.read().result()
        except Exception:
            print(f"Unable to read from image patch at{voxel}!")
            return np.ones(shape)

    def read_voxel(self, voxel, thread_id):
        """
        Reads the intensity value at a given voxel.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel to be read.
        thread_id : Any
            Identifier associated with output.

        Returns
        -------
        int
            Intensity value at voxel.
        """
        return thread_id, int(self.img[voxel].read().result())


class ZarrReader(ImageReader):
    """
    Class that reads image with Zarr library.

    """

    def __init__(self, img_path):
        """
        Constructs a Zarr image reader.

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
        Loads image using the Zarr library.

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
        of the image patch to be read. Default is True.

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
def to_physical(voxel, anisotropy, shift=(0, 0, 0)):
    """
    Converts a voxel coordinate to a physical coordinate by applying the
    anisotropy scaling factors.

    Parameters
    ----------
    voxel : ArrayLike
        Voxel coordinate to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    shift : Tuple[int], optional
        Shift to be applied to "voxel". Default is (0, 0, 0).

    Returns
    -------
    Tuple[float]
        Physical coordinate.
    """
    voxel = voxel[::-1]
    return tuple([voxel[i] * anisotropy[i] - shift[i] for i in range(3)])


def to_voxels(xyz, anisotropy, multiscale=0):
    """
    Converts coordinate from a physical to voxel space.

    Parameters
    ----------
    xyz : ArrayLike
        Physical coordiante to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
        Default is 0.

    Returns
    -------
    Tuple[int]
        Voxel coordinate.
    """
    scaling_factor = 1.0 / 2 ** multiscale
    voxel = [int(scaling_factor * xyz[i] / anisotropy[i]) for i in range(3)]
    return tuple(voxel[::-1])


# --- Helpers ---
def fill_path(img, path, val=-1):
    """
    Fills a given path in a 3D image array with a specified value.

    Parameters
    ----------
    img : numpy.ndarray
        The 3D image array to fill the path in.
    path : iterable
        A list or iterable containing 3D coordinates (x, y, z) representing
        the path.
    val : int, optional
        The value to fill the path with. Default is -1.

    Returns
    -------
    numpy.ndarray
        The modified image array with the path filled with the specified value.
    """
    for xyz in path:
        x, y, z = tuple(np.floor(xyz).astype(int))
        img[x - 2: x + 3, y - 2: y + 3, z - 2: z + 3] = val
    return img


def find_img_path(bucket_name, root_dir, dataset_name):
    """
    Finds the path to an image in a GCS bucket for the dataset given by
    "dataset_name".

    Parameters:
    ----------
    bucket_name : str
        Name of the GCS bucket where the images are stored.
    root_dir : str
        Path to the directory in the GCS bucket where the image is expected to
        be located.
    dataset_name : str
        Name of the dataset to be searched for within the subdirectories.

    Returns:
    -------
    str
        Path of the found dataset subdirectory within the specified GCS bucket.
    """
    for subdir in util.list_gcs_subdirectories(bucket_name, root_dir):
        if dataset_name in subdir:
            return subdir + "whole-brain/fused.zarr/"
    raise f"Dataset not found in {bucket_name} - {root_dir}"


def get_minimal_bbox(voxels, buffer=0):
    """
    Gets the min and max coordinates of a bounding box that contains "voxels".

    Parameters
    ----------
    voxels : numpy.ndarray
        Array containing voxel coordinates.
    buffer : int, optional
        Constant value added/subtracted from the max/min coordinates of the
        bounding box. Default is 0.

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


def get_mip(img, axis=0):
    """
    Compute the maximum intensity projection (MIP) of "img" along "axis".

    Parameters
    ----------
    img : numpy.ndarray
        Image to compute MIP of.
    axis : int, optional
        Projection axis. Default is 0.

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
        Projection axis. Default is 0.

    Returns
    -------
    numpy.ndarray
        MIP of "img".
    """
    mip = get_mip(img, axis=axis)
    return (255 * label2rgb(mip)).astype(np.uint8)


def get_neighbors(voxel, shape):
    """
    Gets the neighbors of a given voxel coordinate.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate in a 3D image.
    shape : Tuplep[int]
        Shape of the 3D image that voxel is contained within.

    Returns
    -------
    List[Tuple[int]]
         Voxel coordinates of the 26 neighbors of the given voxel.
    """
    # Initializations
    x, y, z = voxel
    depth, height, width = shape

    # Iterate over the possible offsets for x, y, and z
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                # Skip the (0, 0, 0) offset, which refers to the voxel itself
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                # Calculate the neighbor's coordinates
                nx, ny, nz = x + dx, y + dy, z + dz

                # Check if the neighbor is within the bounds of the 3D image
                if 0 <= nx < depth and 0 <= ny < height and 0 <= nz < width:
                    neighbors.append((nx, ny, nz))

    return neighbors


def is_contained(bbox, voxel):
    """
    Checks whether a given voxel is contained within the image bounding box
    specified by "bbox".

    Parameters
    ----------
    bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image.
    voxel : Tuple[int]
        Voxel coordinate to be checked.

    Returns
    -------
    bool
        Inidcation of whether "voxel" is contained within the given image
        bounding box.
    """
    above = any([v >= bbox_max for v, bbox_max in zip(voxel, bbox["max"])])
    below = any([v < bbox_min for v, bbox_min in zip(voxel, bbox["min"])])
    return False if above or below else True


def is_list_contained(bbox, voxels):
    """
    Checks whether a list of voxels is contained within a given image bounding
    box.

    Parameters
    ----------
    bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image.
    voxels : List[Tuple[int]]
        List of voxel coordinates to be checked.

    Returns
    -------
    bool
        Indication of whether every element in "voxels" is contained in
        "bbox".
    """
    return all([is_contained(bbox, voxel) for voxel in voxels])


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


def resize(img, new_shape):
    depth, height, width = img.shape
    zoom_factors = np.array(new_shape) / np.array([depth, height, width])
    resized_img = zoom(img, zoom_factors, order=1)
    return resized_img
