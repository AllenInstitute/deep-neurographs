"""
Created on Wed July 2 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that detect merge mistakes.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from deep_neurographs.machine_learning.augmentation import ImageTransforms
from deep_neurographs.skeleton_graph import SkeletonGraph
from deep_neurographs.utils import img_util, swc_util, util


# --- Trainer ---
class MergeDetectionTrainer:

    def __init__(
        self,
        model,
        output_dir,
        batch_size=32,
        lr=1e-3,
        max_epochs=200,
    ):
        # Initializations
        exp_name = "session-" + datetime.today().strftime("%Y%m%d_%H%M")
        log_dir = os.path.join(output_dir, exp_name)
        util.mkdir(log_dir)

        # Instance attributes
        self.batch_size = batch_size
        self.best_f1 = 0
        self.log_dir = log_dir
        self.max_epochs = max_epochs

        self.criterion = nn.BCEWithLogitsLoss()
        self.model = model.to("cuda")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)
        self.writer = SummaryWriter(log_dir=log_dir)

    # --- Core Routines ---
    def run(self, train_dataset, val_dataset):
        """
        Run the full training and validation loop.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Dataset used for training.
        val_dataset : torch.utils.data.Dataset
            Dataset used for validation.

        Returns
        -------
        None
        """
        # Initializations
        exp_name = os.path.basename(os.path.normpath(self.log_dir))
        print("\nExperiment:", exp_name)

        train_dataloader = MergeDetectionGraphDataloader(
            train_dataset, batch_size=self.batch_size
        )
        val_dataloader = MergeDetectionGraphDataloader(
            val_dataset, batch_size=self.batch_size
        )

        # Main
        for epoch in range(self.max_epochs):
            # Train-Validate
            train_stats = self.train_step(train_dataloader, epoch)
            val_stats, new_best = self.validate_step(val_dataloader, epoch)

            # Report reuslts
            print(f"\nEpoch {epoch}: " + ("New Best!" if new_best else " "))
            self.report_stats(train_stats, is_train=True)
            self.report_stats(val_stats, is_train=False)

            # Step scheduler
            self.scheduler.step()

    def train_step(self, train_dataloader, epoch):
        """
        Perform a single training epoch over the provided DataLoader.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        dict
            Dictionary of aggregated training metrics.
        """
        stats = {"f1": None, "precision": [], "recall": [], "loss": []}
        self.model.train()
        for x, y in train_dataloader:
            # Forward pass
            hat_y, loss = self.forward_pass(x, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store stats for tensorboard
            stats["loss"].append(float(to_cpu(loss)))
            for key, value in self.compute_stats(y, hat_y).items():
                stats[key].extend(value)

        # Write stats to tensorboard
        self.update_tensorboard(stats, epoch, "train_")
        return stats

    def validate_step(self, val_dataloader, epoch):
        """
        Perform a full validation loop over the given dataloader.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        tuple
            stats : dict
                Dictionary of aggregated validation metrics.
            is_best : bool
                True if the current F1 score is the best so far.
        """
        stats = {"f1": None, "precision": [], "recall": [], "loss": []}
        with torch.no_grad():
            self.model.eval()
            for x, y in val_dataloader:
                # Run model
                hat_y, loss = self.forward_pass(x, y)

                # Store stats for tensorboard
                stats["loss"].append(float(to_cpu(loss)))
                for key, value in self.compute_stats(y, hat_y).items():
                    stats[key].extend(value)

        # Write stats to tensorboard
        self.update_tensorboard(stats, epoch, "val_")

        # Check for new best
        if stats["f1"] > self.best_f1:
            self.save_model(epoch)
            self.best_f1 = stats["f1"]
            return stats, True
        else:
            return stats, False

    def forward_pass(self, x, y):
        """
        Perform a forward pass through the model and compute loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, D, H, W).
        y : torch.Tensor
            Ground truth labels with shape (B, C, D, H, W).

        Returns
        -------
        tuple
            hat_y : torch.Tensor
                Model predictions.
            loss : torch.Tensor
                Computed loss value.
        """
        x = x.to("cuda", dtype=torch.float32)
        y = y.to("cuda", dtype=torch.float32)
        hat_y = self.model(x)
        loss = self.criterion(hat_y, y)
        return hat_y, loss

    # --- Helpers
    def compute_stats(self, y, hat_y):
        """
        Compute F1 score, precision, and recall for each sample in a batch.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth labels of shape (B, 1, D, H, W) or (B, 1, H, W).
        hat_y : torch.Tensor
            Model predictions of the same shape as ground truth.

        Returns
        -------
        dict
            Dictionary containing lists of per-sample metrics.
        """
        y, hat_y = to_cpu(y, True), to_cpu(hat_y, True)
        stats = {"precision": list(), "recall": list()}
        for i in range(y.shape[0]):
            # Ensure binary format
            gt = (y[i, 0, ...] > 0).astype(np.uint8).flatten()
            pred = (hat_y[i, 0, ...] > 0).astype(np.uint8).flatten()

            # Compute metrics
            stats["precision"].append(precision_score(gt, pred, zero_division=np.nan))
            stats["recall"].append(recall_score(gt, pred, zero_division=np.nan))
        return stats

    def report_stats(self, stats, is_train=True):
        """
        Print a summary of training or validation statistics.

        Parameters
        ----------
        stats : dict
            Dictionary of metric names to values.
        is_train : bool, optional
            Indication of whether "stats" were computed during training step.
            Default is True.

        Returns
        -------
        None
        """
        summary = "   Train: " if is_train else "   Val: "
        for key, value in stats.items():
            summary += f"{key}={value:.4f}, "
        print(summary)

    def save_model(self, epoch):
        """
        Save the current model state to a file.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        None
        """
        date = datetime.today().strftime("%Y%m%d")
        filename = f"UNet3d-{date}-{epoch}-{self.best_f1:.4f}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(self.model.state_dict(), path)

    def update_tensorboard(self, stats, epoch, prefix):
        """
        Log scalar statistics to TensorBoard.

        Parameters
        ----------
        stats : dict
            Dictionary of metric names (str) to lists of values.
        epoch : int
            Current training epoch.
        prefix : str
            Prefix to prepend to each metric name when logging.

        Returns
        -------
        None
        """
        # Compute avg f1 score
        avg_prec = np.nanmean(stats["precision"])
        avg_recall = np.nanmean(stats["recall"])
        stats["f1"] = [2 * avg_prec * avg_recall / (avg_prec + avg_recall)]

        # Write to tensorboard
        for key, value in stats.items():
            stats[key] = np.nanmean(value)
            self.writer.add_scalar(prefix + key, stats[key], epoch)


# --- Dataset ---
class MergeDetectionGraphDataset:

    def __init__(
        self,
        merge_sites_df,
        anisotropy=(1.0, 1.0, 1.0),
        context_radius=200,
        multiscale=0,
        node_spacing=5,
        patch_shape=(84, 84, 84),
        transform=False,
        use_random_sites=True,
    ):
        # Instance attributes
        self.anisotropy = anisotropy
        self.context_radius = context_radius
        self.node_spacing = node_spacing
        self.merge_sites_df = merge_sites_df
        self.multiscale = multiscale
        self.patch_shape = patch_shape
        self.transform = ImageTransforms() if transform else None
        self.use_random_sites = use_random_sites

        # Data structures
        self.img_readers = dict()
        self.gt_graphs = dict()
        self.gt_kdtrees = dict()
        self.merge_graphs = dict()
        self.merge_kdtrees = dict()

    # --- Load Data ---
    def init_graph(self, swc_pointer):
        graph = SkeletonGraph(node_spacing=self.node_spacing)
        graph.load(swc_pointer)
        return graph

    def init_kdtrees(self):
        self.gt_kdtrees = self._init_kdtree(self.gt_graphs)
        self.merge_kdtrees = self._init_kdtree(self.merge_graphs)

    def _init_kdtree(self, graphs):
        for brain_id, graph in graphs.items():
            graph.init_kdtree()

    def load_merge_graphs(self, brain_id, swc_pointer):
        # Load graphs
        graph = self.init_graph(swc_pointer)

        # Filter non-merge components
        idxs = self.merge_sites_df["brain_id"] == brain_id
        merged_segment_ids = self.merge_sites_df["segment_id"][idxs].values
        for swc_id in graph.get_swc_ids():
            segment_id = swc_util.get_segment_id(swc_id)
            if str(segment_id) not in merged_segment_ids:
                component_id = find_key(graph.component_id_to_swc_id, swc_id)
                nodes = graph.get_nodes_with_component_id(component_id)
                graph.remove_nodes_from(nodes)
                graph.irreducible.remove_nodes_from(nodes)

        graph.relabel_nodes()
        self.merge_graphs[brain_id] = graph

    def load_gt_graphs(self, brain_id, img_path, swc_pointer):
        self.img_readers[brain_id] = img_util.init_reader(img_path)
        self.gt_graphs[brain_id] = self.init_graph(swc_pointer)

    # --- Get Examples ---
    def __getitem__(self, idx):
        # Extract site
        brain_id, graph, node, is_positive = self.get_site(idx)
        xyz = graph.node_xyz[node]
        voxel = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)

        # Extract subgraph and image patches centered at site
        subgraph = graph.get_rooted_subgraph(node, self.context_radius)
        img_patch = self.img_readers[brain_id].read(voxel, self.patch_shape)
        img_patch = img_util.resize(img_util.normalize(img_patch), (64, 64, 64))
        label_patch = self.get_label_mask(subgraph)
        patches = np.stack([img_patch, label_patch], axis=0)

        # Apply image augmentation
        if self.transform:
            _, patches = self.transform(idx, patches)
        return patches, subgraph, int(is_positive)

    def get_site(self, idx):
        brain_id = self.merge_sites_df["brain_id"][idx]
        is_positive = np.random.random() > 0.5
        if is_positive:
            graph = self.merge_graphs[brain_id]
        else:
            if np.random.random() > 0.5 and self.use_random_sites:
                return self.get_random_site()
            else:
                graph = self.gt_graphs[brain_id]
        xyz = self.merge_sites_df["xyz"][idx]
        node = graph.query_node(xyz)
        return brain_id, graph, node, is_positive

    def get_random_site(self):
        brain_id = util.sample_once(list(self.gt_graphs.keys()))
        graph = self.gt_graphs[brain_id]
        node = np.random.randint(0, graph.number_of_nodes())
        return brain_id, graph, node, False

    def get_label_mask(self, subgraph):
        # Initializations
        label_mask = np.zeros(self.patch_shape)
        xyz = subgraph.node_xyz[0]
        center = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)

        # Populate label mask
        for i in subgraph.nodes:
            xyz = subgraph.node_xyz[i]
            voxel = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)
            voxel = shift_voxel(voxel, center, self.patch_shape)
            if img_util.is_contained(voxel, self.patch_shape, buffer=3):
                i, j, k = voxel
                label_mask[i-3:i+3, j-3:j+3, k-3:k+3] = 1
        return img_util.resize(label_mask, (64, 64, 64)).astype(int)

    # --- Helpers ---
    def __len__(self):
        return len(self.merge_sites_df)


# --- Dataloader ---
class MergeDetectionGraphDataloader:
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.
    """

    def __init__(self, dataset, batch_size=32):
        # Instance attributes
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):
        for idx in range(0, len(self.dataset), self.batch_size):
            yield self._load_batch(idx)

    def _load_batch(self, start_idx):
        # Compute batch size
        n_remaining_examples = len(self.dataset) - start_idx
        batch_size = min(self.batch_size, n_remaining_examples)

        # Generate batch
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for idx_shift in range(batch_size):
                idx = start_idx + idx_shift
                threads.append(executor.submit(self.dataset.__getitem__, idx))

            # Process results
            patches = np.zeros((batch_size, 2, 64, 64, 64), dtype=np.float32)
            labels = np.zeros((batch_size, 1), dtype=np.float32)
            for i, thread in enumerate(as_completed(threads)):
                patch, _, label = thread.result()
                patches[i] = patch
                labels[i] = label
        return to_tensor(patches), to_tensor(labels)


# -- Helpers --
def find_key(my_dict, target_value):
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None


def shift_voxel(voxel, center, patch_shape):
    voxel = [v - c + s // 2 for v, c, s in zip(voxel, center, patch_shape)]
    return tuple(voxel)


def to_cpu(tensor, to_numpy=False):
    """
    Move PyTorch tensor to the CPU and optionally convert it to a NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to move to CPU.
    to_numpy : bool, optional
        If True, converts the tensor to a NumPy array. Default is False.

    Returns
    -------
    torch.Tensor or np.ndarray
        Tensor or array on CPU.
    """
    if to_numpy:
        return np.array(tensor.detach().cpu())
    else:
        return tensor.detach().cpu()


def to_tensor(arr):
    """
    Converts a numpy array to a tensor.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Array converted to tensor.

    """
    return torch.tensor(arr, dtype=torch.float32)
