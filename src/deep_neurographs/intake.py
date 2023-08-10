"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds graph for postprocessing with GNN.

"""

import numpy as np
import torch
from deep_neurographs import graph_classes as graph_class
from deep_neurographs import neurograph as ng
from deep_neurographs import s3_utils, swc_utils, utils
from torch_geometric.data import Data


# --- Build graph ---
def generate_immutables(
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


def build_immutable_from_local(
    neurograph, swc_dir, anisotropy=[1.0, 1.0, 1.0]
):
    """
    To do...
    """
    pass


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


"""
def build_supergraph(
    bucket,
    swc_path,
    access_key_id=None,
    secret_access_key=None,
    anisotropy=[1.0, 1.0, 1.0],
):
    s3_client = s3_utils.init_session(
        access_key_id=access_key_id, secret_access_key=secret_access_key
    )
    graph = graph_class.SuperGraph()
    graph = create_nodes_from_swc(
        graph,
        bucket,
        s3_client,
        swc_path,
        anisotropy=anisotropy,
    )
    graph.create_edges()
    return graph

def create_nodes_from_swc(
    graph,
    bucket,
    s3_client,
    swc_path,
    anisotropy=[1.0, 1.0, 1.0],
):
    file_keys = s3_utils.listdir(bucket, swc_path, s3_client, ext=".swc")
    for node_id, file_key in enumerate(file_keys):
        # Parse and add node
        raw_swc = s3_utils.read_from_s3(bucket, file_key, s3_client)
        swc_dict = swc_utils.parse(raw_swc, anisotropy=anisotropy)
        graph.add_node_from_swc(node_id, swc_dict)

        # Store upd_id --> cur_id and xyz --> upd_id
        f = file_key.split("/")[-1]
        graph.old_node_ids[node_id] = int(f.split(".")[0])
        graph.upd_xyz_to_id(node_id)
    return graph
    """
