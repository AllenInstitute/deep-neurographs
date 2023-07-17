"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds graph for postprocessing with GNN.

"""

import numpy as np


def generate_skel_features(swc_dict):
    mean_radius = np.mean(swc_dict["radius"])
    mean_xyz = np.mean(swc_dict["xyz"], axis=0)
    return np.concatenate((mean_radius, mean_xyz), axis=None)


def generate_img_features():
    pass


def generate_pointcloud_features():
    pass
