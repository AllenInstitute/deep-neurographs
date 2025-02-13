"""
Created on Tue Jan 21 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for applying image augmentation during training.

"""

from scipy.ndimage import rotate, zoom

import numpy as np
import random
import torch


# --- Geometric Transforms ---
class GeometricTransforms:
    """
    Class that applies a series of geometric transforms to a 3D image and
    label patch.

    """
    def __init__(self):
        self.transforms = [
            RandomFlip3D(),
            RandomRotation3D(),
            RandomScale3D()
        ]

    def __call__(self, img_patch, label_patch):
        # Apply transforms
        for transform in self.transforms:
            img_patch, label_patch = transform(img_patch, label_patch)
        return img_patch, label_patch

class RandomFlip3D:
    """
    Randomly flip a 3D image along one or more axes.

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

    def __call__(self, img_patch, label_patch):
        """
        Applies random flipping to the input 3D image.

        Parameters
        ----------
        img_patch : numpy.ndarray
            Image patch to be flipped.
        label_patch : numpy.ndarray
            Segmentation patch to be flipped.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Flipped 3D image and segmentation patch.

        """
        for axis in self.axes:
            if random.random() > 0.5:
                img_patch = np.flip(img_patch, axis=axis)
                label_patch = np.flip(label_patch, axis=axis)
        return img_patch, label_patch


class RandomRotation3D:
    """
    Applies random rotation to a 3D image along a randomly chosen axis.

    """

    def __init__(self, angles=(-45, 45), axes=((0, 1), (0, 2), (1, 2))):
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

    def __call__(self, img_patch, label_patch):
        """
        Rotates the input 3D image.

        Parameters
        ----------
        img_patch : numpy.ndarray
            Image patch to be rotated.
        label_patch : numpy.ndarray
            Segmentation patch to be rotated.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Rotated 3D image and segmentation patch.

        """
        for axes in self.axes:
            angle = random.uniform(*self.angles)
            img_patch = rotate3d(img_patch, angle, axes)
            label_patch = rotate3d(label_patch, angle, axes)
        return img_patch, label_patch


class RandomScale3D:
    """
    Applies random scaling to a 3D image along each axis.

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

    def __call__(self, img_patch, label_patch):
        """
        Applies random rescaling to the input 3D image.

        Parameters
        ----------
        img_patch : numpy.ndarray
            Image patch to be rescaled.
        label_patch : numpy.ndarray
            Segmentation patch to be rescaled.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Rescaled 3D image and segmentation patch.

        """
        # Sample new image shape
        alpha = np.random.uniform(self.scale_range[0], self.scale_range[1])
        new_shape = (
            int(img_patch.shape[0] * alpha),
            int(img_patch.shape[1] * alpha),
            int(img_patch.shape[2] * alpha),
        )

        # Compute the zoom factors
        shape = img_patch.shape
        zoom_factors = [
            new_dim / old_dim for old_dim, new_dim in zip(shape, new_shape)
        ]

        # Rescale images
        img_patch = zoom(img_patch, zoom_factors, order=3)
        label_patch = zoom(label_patch, zoom_factors, order=3)
        return img_patch, label_patch


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
        return np.clip(img * factor, img.min(), img.max())


class RandomNoise3D:
    """
    Adds random Gaussian noise to a 3D image.

    """

    def __init__(self, mean=0.0, std=0.025):
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
        self.mean = mean
        self.std = std

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
        noise = np.random.normal(self.mean, self.std, img.shape)
        return img + noise


# --- Helpers ---
def rotate3d(img_patch, angle, axes):
    img_patch = rotate(
        img_patch,
        angle,
        axes=axes,
        mode="grid-mirror",
        reshape=False,
        order=0,
    )
    return img_patch
