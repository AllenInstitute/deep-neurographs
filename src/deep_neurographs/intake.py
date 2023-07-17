"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds graph for postprocessing with GNN.

"""

from time import time
from scipy.spatial import KDTree

from deep_neurographs import feature_extraction as extractor
from deep_neurographs import s3_utils, swc_utils, utils
from deep_neurographs import graph_classes as gclass

"""
Sketch for build_graph

    # X - read files

    # -- Node creation --
    # X - init graph
    # X - init dict for (x, y, z) --> new id
    # X - init upd_obj_id dict
    # X - loop through swc files
    # X - create node <-- function call
    # X - assign new id
    # X - generate features <-- function call
    # X - store skeleton in sparse dict (for edge creation)
    
    # -- Edge creation --
    # O(dn) search <-- function call

    # -- Generate ground truth connectivity --
    # read mistake log
    # generate target adjacency matrix
"""


def build_graph(
    bucket,
    label_path,
    swc_path,
    mistake_log_path,
    access_key_id=None,
    secret_access_key=None,
    anisotropy=[1.0, 1.0, 1.0],
):
    # Initialize s3 session
    s3_client = s3_utils.init_session(
        access_key_id=None, secret_access_key=None
    )

    # Build graph
    graph = create_nodes(
        bucket,
        swc_path,
        label_path,
        s3_client,
        anisotropy=anisotropy,
    )
    return graph
    #edge_features = create_edges(xyz_to_id)

    # Create PyG graph


def create_nodes(
    bucket, swc_path, label_path, s3_client, anisotropy=[1.0, 1.0, 1.0]
):
    graph = gclass.SuperGraph()
    file_keys = s3_utils.listdir(bucket, swc_path, s3_client, ext=".swc")
    for node_id, key in enumerate(file_keys):
        # Read and process data
        raw_swc = s3_utils.read_from_s3(bucket, key, s3_client)
        swc_dict = swc_utils.parse(raw_swc, anisotropy=anisotropy)

        # Populate graph
        graph.add_node(node_id)
        graph.set_node_attribute(node_id, "radius", swc_dict["radius"])
        graph.set_node_attribute(node_id, "subnodes", swc_dict["subnodes"])
        graph.set_node_attribute(node_id, "xyz", swc_dict["xyz"])

        leafs, junctions = swc_utils.extract_topo_nodes(swc_dict["subnodes"], swc_dict["parents"])
        graph.set_node_attribute(node_id, "leafs", leafs)
        graph.set_node_attribute(node_id, "junctions", junctions)

        # Feature extraction
        skel_features = extractor.generate_skel_features(swc_dict)
        graph.add_node_feature(skel_features)

        # Store upd_id --> cur_id and xyz --> upd_id
        f = key.split("/")[-1]
        graph.old_node_ids[node_id] = int(f.split(".")[0])
        graph.upd_xyz_to_id(node_id)
        #= upd_dict(xyz_to_id, swc_dict["xyz"], node_id)

    return graph


def upd_dict(my_dict, keys, scalar):
    my_dict.update(dict(zip_broadcast(keys, scalar)))
    return my_dict


def create_edges(xyz_to_id):
    kdtree = KDTree(list(xyz_to_id.keys()))
    t, unit = utils.time_writer(time() - t0)


def get_target_labels(bucket, mistake_log_path):
    mistake_log = s3_utils.read_from_s3(bucket, mistake_log_path)
