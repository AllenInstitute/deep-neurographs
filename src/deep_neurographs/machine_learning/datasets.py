"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom datasets for training deep learning models.

"""

import numpy as np
import torchio as tio
from torch.utils.data import Dataset


# Custom datasets
class ProposalDataset(Dataset):
    """
    Custom dataset that contains feature vectors that correspond to edge
    proposals.

    """

    def __init__(
        self, inputs, targets, search_radius=10, transform=False, lengths=[]
    ):
        """
        Constructs ProposalDataset object.

        Parameters
        ----------
        inputs : np.array
            Feature matrix where each row corresponds to the feature vector of
            an edge proposal.
        targets : np.array
            Binary vector where each entry indicates whether an edge proposal
            should be added or omitted from a reconstruction.

        Returns
        -------
        None

        """
        self.inputs = inputs.astype(np.float32)
        self.targets = reformat(targets)
        self.lengths = lengths
        self.transform = transform

    def __len__(self):
        """
        Computes number of examples in dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of examples in dataset.

        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Gets example (i.e. input and label) corresponding to "idx".

        Parameters
        ----------
        idx : int
            Index of example to be returned.

        Returns
        -------
        dict
            Example corresponding to "idx".

        """
        inputs_i = self.inputs[idx]
        if self.transform:
            if np.random.random() > 0.6:
                p = 100 * np.random.random()
                inputs_i[0] = np.percentile(self.lengths, p)
        return {"inputs": inputs_i, "targets": self.targets[idx]}


class ImgProposalDataset(Dataset):
    """
    Custom dataset that contains image chunks that correspond to edge
    proposals.

    """

    def __init__(self, inputs, targets, transform=True):
        """
        Constructs ImgProposalDataset object.

        Parameters
        ----------
        inputs : numpy.array
            Feature tensor where each submatrix corresponds to an image chunk
            that contains an edge proposal. Note that the midpoint of the edge
            proposal is the center point of the chunk.
        targets : np.array
            Binary vector where each entry indicates whether an edge proposal
            should be added or omitted from a reconstruction.
        transform : bool, optional
            Indication of whether to apply data augmentation to the inputs.
            The default is True.

        Returns
        -------
        None

        """
        self.inputs = inputs.astype(np.float32)
        self.targets = reformat(targets)
        self.transform = AugmentImages() if transform else None

    def __len__(self):
        """
        Computes number of examples in dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of examples in dataset.

        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Gets example (i.e. input and label) corresponding to "idx".

        Parameters
        ----------
        idx : int
            Index of example to be returned.

        Returns
        -------
        dict
            Example corresponding to "idx".

        """
        if self.transform:
            inputs = self.transform.run(self.inputs[idx])
        else:
            inputs = self.inputs[idx]
        return {"inputs": inputs, "targets": self.targets[idx]}


class MultiModalDataset(Dataset):
    """
    Custom multimodal dataset that contains both feature vectors and image
    chunks that correspond to edge proposals.

    """

    def __init__(self, inputs, targets, transform=True):
        """
        Constructs MultiModalDataset object.

        Parameters
        ----------
        inputs : dict
            Feature dictionary where each key-value is the type of feature and
            corresponding value. The keys of this dictionary are (1) "imgs" and
            (2) "features" which correspond to a (1) feature tensor containing
            image chunks and (2) feature vector.
        targets : np.array
            Binary vector where each entry indicates whether an edge proposal
            should be added or omitted from a reconstruction.
        transform : bool, optional
            Indication of whether to apply augmentation to "inputs["imgs"]".
            The default is True.

        Returns
        -------
        None

        """
        self.img_inputs = inputs["imgs"].astype(np.float32)
        self.feature_inputs = inputs["features"].astype(np.float32)
        self.targets = reformat(targets)
        self.transform = AugmentImages() if transform else None

    def __len__(self):
        """
        Computes number of examples in dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of examples in dataset.

        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Gets example (i.e. input and label) corresponding to "idx".

        Parameters
        ----------
        idx : int
            Index of example to be returned.

        Returns
        -------
        dict
            Example corresponding to "idx".

        """
        if self.transform:
            img_inputs = self.transform.run(self.img_inputs[idx])
        else:
            img_inputs = self.img_inputs[idx]
        inputs = [self.feature_inputs[idx], img_inputs]
        return {"inputs": inputs, "targets": self.targets[idx]}


# Augmentation
class AugmentImages:
    """
    Applies augmentation to an image chunk.

    """

    def __init__(self):
        """
        Constructs an AugmentImages object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.transform = tio.Compose(
            [
                tio.RandomBlur(std=(0, 0.4)),
                tio.RandomNoise(std=(0, 0.0125)),
                tio.RandomFlip(axes=(0, 1, 2)),
                tio.RandomAffine(
                    degrees=20, scales=(0.8, 1), image_interpolation="nearest"
                ),
            ]
        )

    def run(self, arr):
        """
        Runs an image through the data augmentation pipeline.

        Parameters
        ----------
        arr : torch.array
            Array that contains an image chunk.

        Returns
        -------
        torch.array
            Transformed array after being run through augmentation pipeline.

        """
        return self.transform(arr)


def get_lengths(neurograph):
    lengths = []
    for edge in neurograph.proposals.keys():
        lengths.append(neurograph.proposal_length(edge))
    return lengths


# -- utils --
def reformat(arr):
    """
    Reformats a label vector for training by adding a dimension and casting it
    to float32.

    Parameters
    ----------
    arr : numpy.array
        Label vector.

    Returns
    -------
    numpy.arr
        Reformatted label vector.

    """
    return np.expand_dims(arr, axis=1).astype(np.float32)
