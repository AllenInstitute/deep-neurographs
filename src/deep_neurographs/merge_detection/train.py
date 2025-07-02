"""
Created on Wed July 2 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that detect merge mistakes.

"""


# --- Custom Trainer ---


# --- Custom Dataset ---
class GraphDataset:

    def __init__(self, merge_sites_df):
        # Instance attributes
        self.imgs = dict()
        self.merge_sites_df = merge_sites_df

    def load_fragment_graphs(self):
        pass

    def load_gt_graphs(self, brain_id, gt_pointer, img_path):
        # Store image
        self.imgs[brain_id] = img_path

        # Load graphs

# --- Custom Dataloader ---


# -- Helpers --