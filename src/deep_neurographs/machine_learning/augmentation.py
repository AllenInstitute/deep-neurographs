"""
Created on Tue Jan 21 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for applying image augmentation during training.

"""

from scipy.ndimage import rotate, zoom

import numpy as np
import random
import torchvision.transforms as transforms


class ImageTransforms:
    """
    Class that applies a sequence of transforms to a 3D image and
    segmentation patch.

    """
    def __init__(self):
        """
        Initializes a GeometricTransforms instance that applies geomtery-based
        augmentation to an image and segmentation patch.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Instance attributes
        self.geometric_transforms = [
            RandomFlip3D(),
            RandomRotation3D(),
        ]
        self.intensity_transforms = transforms.Compose(
                [RandomContrast3D(), RandomNoise3D()]
            )

    def __call__(self, patch_id, patches):
        """
        Applies geometric transforms to the input image and segmentation
        patch.

        Parameters
        ----------
        patch_id : hashable
            Unique identifier for patches.
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where "patches[0, ...]" is from
            the input image and "patches[1, ...]" is from the segmentation.

        Returns
        -------
        numpy.ndarray
            Transformed 3D image and segmentation patch.

        """
        # Geometric transforms
        for transform in self.geometric_transforms:
            patches = transform(patches)

        # Intensity transforms
        patches[0, ...] = self.intensity_transforms(patches[0, ...])
        return patch_id, patches


# --- Geometric Transforms ---
class RandomFlip3D:
    """
    Randomly flips a 3D image along one or more axes.

    """

    def __init__(self, axes=(0, 1, 2)):
        """
        Initializes a RandomFlip3D transformer.

        Parameters
        ----------
        axes : Tuple[float], optional
            Tuple of integers representing the axes along which to flip the
            image. The default is (0, 1, 2).

        Returns
        -------
        None

        """
        self.axes = axes

    def __call__(self, patches):
        """
        Applies random flipping to the input image and segmentation patch.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where "patches[0, ...]" is from
            the input image and "patches[1, ...]" is from the segmentation.

        Returns
        -------
        numpy.ndarray
            Flipped 3D image and segmentation patch.

        """
        for axis in self.axes:
            if random.random() > 0.5:
                patches[0, ...] = np.flip(patches[0, ...], axis=axis)
                patches[1, ...] = np.flip(patches[1, ...], axis=axis)
        return patches


class RandomRotation3D:
    """
    Applies random rotation along a randomly chosen axis.

    """

    def __init__(self, angles=(-90, 90), axes=((0, 1), (0, 2), (1, 2))):
        """
        Initializes a RandomRotation3D transformer.

        Parameters
        ----------
        angles : Tuple[int], optional
            Maximum angle of rotation. The default is (-45, 45).
        axis : Tuple[Tuple[int]], optional
            Axes to apply rotation.

        Returns
        -------
        None

        """
        self.angles = angles
        self.axes = axes

    def __call__(self, patches):
        """
        Rotates the input image and segmentation patch.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where "patches[0, ...]" is from
            the input image and "patches[1, ...]" is from the segmentation.

        Returns
        -------
        numpy.ndarray
            Rotated 3D image and segmentation patch.

        """
        for axes in self.axes:
            if random.random() > 0.4:
                angle = random.uniform(*self.angles)
                patches[0, ...] = rotate3d(patches[0, ...], angle, axes)
                patches[1, ...] = rotate3d(patches[1, ...], angle, axes)
        return patches


class RandomScale3D:
    """
    Applies random scaling along each axis.

    """

    def __init__(self, scale_range=(0.9, 1.1)):
        """
        Initializes a RandomScale3D transformer.

        Parameters
        ----------
        scale_range : Tuple[float], optional
            Range of scaling factors. The default is (0.9, 1.1).

        Returns
        -------
        None

        """
        self.scale_range = scale_range

    def __call__(self, patches):
        """
        Applies random rescaling to the input 3D image.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where "patches[0, ...]" is from
            the input image and "patches[1, ...]" is from the segmentation.

        Returns
        -------
        numpy.ndarray
            Rescaled 3D image and segmentation patch.

        """
        # Sample new image shape
        alpha = np.random.uniform(self.scale_range[0], self.scale_range[1])
        new_shape = (
            int(patches.shape[1] * alpha),
            int(patches.shape[2] * alpha),
            int(patches.shape[3] * alpha),
        )

        # Compute the zoom factors
        shape = patches.shape[1:]
        zoom_factors = [
            new_dim / old_dim for old_dim, new_dim in zip(shape, new_shape)
        ]

        # Rescale images
        patches[0, ...] = zoom(patches[0, ...], zoom_factors, order=3)
        patches[1, ...] = zoom(patches[1, ...], zoom_factors, order=3)
        return patches


# --- Intensity Transforms ---
class RandomContrast3D:
    """
    Adjusts the contrast of a 3D image by scaling voxel intensities.

    """

    def __init__(self, factor_range=(0.8, 1.2)):
        """
        Initializes a RandomContrast3D transformer.

        Parameters
        ----------
        factor_range : Tuple[float], optional
            Tuple of integers representing the range of contrast factors. The
            default is (0.8, 1.1).

        Returns
        -------
        None

        """
        self.factor_range = factor_range

    def __call__(self, img):
        """
        Applies contrast to the input 3D image.

        Parameters
        ----------
        img : numpy.ndarray
            Image to which contrast will be added.

        Returns
        -------
        numpy.ndarray
            Contrasted 3D image.

        """
        factor = random.uniform(*self.factor_range)
        return np.clip(img * factor, 0, 1)


class RandomNoise3D:
    """
    Adds random Gaussian noise to a 3D image.

    """

    def __init__(self, max_std=0.06):
        """
        Initializes a RandomNoise3D transformer.

        Parameters
        ----------
        mean : float, optional
            Mean of the Gaussian noise distribution. The default is 0.0.
        std : float, optional
            Standard deviation of the Gaussian noise distribution. The default
            is 0.025.

        Returns
        -------
        None

        """
        self.max_std = max_std

    def __call__(self, img):
        """
        Adds Gaussian noise to the input 3D image.

        Parameters
        ----------
        img : np.ndarray
            Image to which noise will be added.

        Returns
        -------
        numpy.ndarray
            Noisy 3D image.

        """
        std = self.max_std * random.random()
        noise = np.random.normal(0, std, img.shape)
        return img + noise


# --- Helpers ---
def rotate3d(img_patch, angle, axes):
    """
    Rotates a 3D image patch around the specified axes by a given angle.

    Parameters
    ----------
    img_patch : numpy.ndarray
        Image to be rotated.
    angle : float
        Angle (in degrees) by which to rotate the image patch around the
        specified axes.
    axes : Tuple[int]
        Tuple representing the two axes of rotation.

    Returns
    -------
    numpy.ndarray
        Rotated 3D image patch.

    """
    img_patch = rotate(
        img_patch,
        angle,
        axes=axes,
        mode="grid-mirror",
        reshape=False,
        order=0,
    )
    return img_patch
