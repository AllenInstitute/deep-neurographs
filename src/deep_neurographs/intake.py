"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds graph for postprocessing with GNN.

"""

from deep_neurographs import graph_classes as graph_class
from deep_neurographs import s3_utils, swc_utils


def build_graph(
    bucket,
    swc_path,
    access_key_id=None,
    secret_access_key=None,
    anisotropy=[1.0, 1.0, 1.0],
):
    # Initialize s3 session
    s3_client = s3_utils.init_session(
        access_key_id=access_key_id, secret_access_key=secret_access_key
    )

    # Build supergraph
    graph = graph_class.SuperGraph()
    graph = create_nodes_from_swc(
        graph, bucket, s3_client, swc_path, anisotropy=anisotropy,
    )
    graph.create_edges()
    return graph


def create_nodes_from_swc(
    graph, bucket, s3_client, swc_path, anisotropy=[1.0, 1.0, 1.0],
):
    """
    """
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


def create_nodes_from_mask():
    """
    """
    pass


def get_target_edges(supergraph, bucket, file_key, access_key_id=None, secret_access_key=None):
    s3_client = s3_utils.init_session(
        access_key_id=access_key_id, secret_access_key=secret_access_key
    )
    mistake_log = s3_utils.read_from_s3(bucket, file_key, s3_client)
    # convert edges to new ids
    return mistake_log