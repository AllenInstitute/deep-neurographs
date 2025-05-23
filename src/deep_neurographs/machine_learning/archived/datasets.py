"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom datasets for training deep learning models.

"""

from torch.utils.data import Dataset as TorchDataset

import numpy as np

from deep_neurographs.machine_learning import feature_generation


# Wrapper
def init(neurograph, features, sample_ids=None):
    """
    Initializes a dataset that can be used to train a machine learning model.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that dataset is built from.
    features : dict
        Feature vectors corresponding to branches such that the keys are
        "proposals" and "branches". The values are a dictionary containing
        different types of features for edges and branches.
    is_multimodal : bool, optional
        Indication of whether model is multimodal. The default is False.
    sample_ids : list[str]
        ...

    Returns
    -------
    GraphDataset
        Custom dataset.

    """
    # Extract features
    x_proposals, y_proposals, idxs_proposals = feature_generation.get_matrix(
        neurograph, features["proposals"], sample_ids=sample_ids
    )

    # Initialize dataset
    proposals = list(features["proposals"]["skel"].keys())
    dataset = Dataset(proposals, x_proposals, y_proposals, idxs_proposals)
    return dataset


class Dataset:
    """
    Dataset class that contains feature vectors of edge proposals. The feature
    vectors may be either unimodal or multimodal.

    """

    def __init__(self, proposals, x_proposals, y_proposals, idxs_proposals):
        """
        Constructs a Dataset object.

        Parameters
        ----------
        computation_graph : networkx.Graph
            Graph used by gnn to classify proposals.
        proposals : list
            List of proposals to be classified.
        x_proposals : numpy.ndarray
            Feature matrix generated from "proposals" in "computation_graph".
        y_proposals : numpy.ndarray
            Ground truth of proposals (i.e. accept or reject).
        idxs_proposals : dict
            Dictionary that maps "proposals" to an index that represents the
            proposal's position in "x_proposals".

        Returns
        -------
        None

        """
        # Conversion idxs
        self.block_to_idxs = idxs_proposals["block_to_idxs"]
        self.idxs_proposals = idxs_proposals
        self.proposals = proposals

        # Features
        self.data = ProposalDataset(x=x_proposals, y=y_proposals)


class ProposalDataset(TorchDataset):
    """
    Custom dataset that contains feature vectors that correspond to edge
    proposals.

    """

    def __init__(self, x, y):
        """
        Constructs ProposalDataset object.

        Parameters
        ----------
        x : np.array
            Feature matrix where each row corresponds to the feature vector of
            a proposal.
        y : np.array
            Ground truth of proposals (i.e. accept or reject).

        Returns
        -------
        None

        """
        self.x = x.astype(np.float32)
        self.y = reformat(y)

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
        return len(self.y)

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
        return {"inputs": self.x[idx], "targets": self.y[idx]}


class MultiModalDataset(Dataset):
    """
    Custom multimodal dataset that contains both feature vectors and image
    chunks that correspond to edge proposals.

    """

    def __init__(self, x, y, transform=True):
        """
        Constructs MultiModalDataset object.

        Parameters
        ----------
        x : dict
            Feature dictionary where each key-value is the type of feature and
            corresponding value. The keys of this dictionary are (1) "imgs" and
            (2) "features" which correspond to a (1) feature tensor containing
            image chunks and (2) feature vector.
        y : np.array
            Binary vector where each entry indicates whether an edge proposal
            should be added or omitted from a reconstruction.
        transform : bool, optional
            Indication of whether to apply augmentation to "inputs["imgs"]".
            The default is True.

        Returns
        -------
        None

        """
        self.x_imgs = x["imgs"].astype(np.float32)
        self.x_features = x["features"].astype(np.float32)
        self.y = reformat(y)
        self.transform = None

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
        return len(self.y)

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
            x_img = self.transform.run(self.x_imgs[idx])
        else:
            x_img = self.x_imgs[idx]
        inputs = [self.feature_inputs[idx], x_img]
        return {"inputs": inputs, "targets": self.targets[idx]}


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
