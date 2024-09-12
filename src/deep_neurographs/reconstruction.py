"""
Created on Sat Dec 12 17:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

import os

import networkx as nx
import numpy as np

from deep_neurographs import graph_utils as gutils
from deep_neurographs import swc_utils, utils



def save_prediction(
    neurograph, accepted_proposals, output_dir, save_swcs=False
):
    # Initializations
    connections_path = os.path.join(output_dir, "connections.txt")
    corrections_dir = os.path.join(output_dir, "corrections")
    swc_zip_path = os.path.join(output_dir, "corrected-processed-swcs.zip")
    utils.mkdir(corrections_dir, delete=True)

    # Write Result
    n_swcs = gutils.count_components(neurograph)
    save_connections(neurograph, connections_path)
    if save_swcs:
        neurograph.to_zipped_swcs(swc_zip_path)
        save_corrections(neurograph, accepted_proposals, corrections_dir)
    else:
        print(f"Result contains {n_swcs} swcs!")


def save_corrections(neurograph, accepted_proposals, output_dir):
    for cnt, (i, j) in enumerate(accepted_proposals):
        # Info
        color = f"1.0 1.0 1.0"
        filename = f"merge-{cnt + 1}.swc"
        path = os.path.join(output_dir, filename)

        # Save
        xyz_i = neurograph.nodes[i]["xyz"]
        xyz_j = neurograph.nodes[j]["xyz"]
        swc_utils.save_edge(path, xyz_i, xyz_j, color=color, radius=3)


def save_connections(neurograph, path):
    """
    Saves predicted connections between connected components in a txt file.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph built from predicted swc files.
    accepted_proposals : list[frozenset]
        List of accepted edge proposals where each entry is a frozenset that
        consists of the nodes corresponding to a predicted connection.
    path : str
        Path that output is written to.

    Returns
    -------
    None

    """
    with open(path, "w") as f:
        for swc_id_i, swc_id_j in neurograph.merged_ids:
            f.write(f"{swc_id_i}, {swc_id_j}" + "\n")
