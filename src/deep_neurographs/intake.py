"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds graph for postprocessing with GNN.

"""

import os

import numpy as np
import torch
from torch_geometric.data import Data

from deep_neurographs import neurograph as ng
from deep_neurographs import s3_utils, swc_utils, utils


# --- Build graph ---
def build_neurograph(
    swc_dir,
    anisotropy=[1.0, 1.0, 1.0],
    bucket=None,
    access_key_id=None,
    secret_access_key=None,
):
    neurograph = ng.NeuroGraph()
    if bucket is not None:
        neurograph = init_immutables_from_s3(
            neurograph,
            bucket,
            swc_dir,
            anisotropy=anisotropy,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )
    else:
        neurograph = init_immutables_from_local(
            neurograph, swc_dir, anisotropy=anisotropy
        )
    neurograph.generate_mutables()
    return neurograph


def init_immutables_from_s3(
    neurograph,
    bucket,
    swc_dir,
    anisotropy=[1.0, 1.0, 1.0],
    access_key_id=None,
    secret_access_key=None,
):
    """
    To do...
    """
    s3_client = s3_utils.init_session(
        access_key_id=access_key_id, secret_access_key=secret_access_key
    )
    file_keys = s3_utils.listdir(bucket, swc_dir, s3_client, ext=".swc")
    for swc_id, file_key in enumerate(file_keys):
        raw_swc = s3_utils.read_from_s3(bucket, file_key, s3_client)
        swc_dict = swc_utils.parse(raw_swc, anisotropy=anisotropy)
        neurograph.generate_immutables(swc_id, swc_dict)
    return neurograph


def init_immutables_from_local(
    neurograph, swc_dir, anisotropy=[1.0, 1.0, 1.0]
):
    """
    To do...
    """
    for swc_id, file_key in enumerate(utils.listdir(swc_dir, ext=".swc")):
        raw_swc = swc_utils.read_swc(os.path.join(swc_dir, file_key))
        swc_dict = swc_utils.parse(raw_swc, anisotropy=anisotropy)
        neurograph.generate_immutables(swc_id, swc_dict)
    return neurograph


# --- Generate training data ---
def init_data(
    supergraph,
    node_features,
    edge_features,
    bucket,
    file_key,
    access_key_id=None,
    secret_access_key=None,
):
    """
    To do...
    """
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(list(supergraph.edges()), dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    edge_label_index, mistake_log = get_target_edges(
        supergraph,
        edge_index.tolist(),
        bucket,
        file_key,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )
    data = Data(
        x=x,
        edge_index=edge_index.t().contiguous(),
        edge_label_index=edge_label_index,
        edge_attr=edge_features,
    )
    return data, mistake_log


def get_target_edges(
    supergraph,
    edges,
    bucket,
    file_key,
    access_key_id=None,
    secret_access_key=None,
):
    """
    To do...
    """
    s3_client = s3_utils.init_session(
        access_key_id=access_key_id, secret_access_key=secret_access_key
    )
    hash_table = read_mistake_log(bucket, file_key, s3_client)
    target_edges = torch.zeros((len(edges)))
    cnt = 0
    for i, e in enumerate(edges):
        e1, e2 = get_old_edge(supergraph, e)
        if utils.check_key(hash_table, e1) or utils.check_key(hash_table, e2):
            target_edges[i] = 1
            cnt += 1
    print("Number of mistakes:", len(hash_table))
    print("Number of hits:", cnt)
    return torch.tensor(target_edges), hash_table


def get_old_edge(supergraph, edge):
    """
    To do...
    """
    id_0 = supergraph.old_node_ids[edge[0]]
    id_1 = supergraph.old_node_ids[edge[1]]
    return (id_0, id_1), (id_1, id_0)


def read_mistake_log(bucket, file_key, s3_client):
    """
    To do...
    """
    hash_table = dict()
    mistake_log = s3_utils.read_from_s3(bucket, file_key, s3_client)
    for entry in mistake_log:
        entry = entry.replace("[", "")
        entry = entry.replace("]", "")
        entry = entry.split(",")
        entry = list(map(float, entry))

        edge = (int(entry[0]), int(entry[1]))
        xyz_coords = (entry[2:5], entry[5:])
        hash_table[edge] = xyz_coords
    return hash_table
