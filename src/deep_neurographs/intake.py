"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds graph for postprocessing with GNN.

"""

from deep_neurographs import feature_extraction as extractor
from deep_neurographs import s3_utils, swc_utils
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

    # Build supergraph
    graph = gclass.SuperGraph()
    graph = create_nodes_from_swc(
        graph,
        bucket,
        s3_client,
        swc_path,
        anisotropy=anisotropy,
    )
    graph.create_edges()

    # Create PyG graph
    
    return graph


def create_nodes_from_swc(
    graph, bucket, s3_client, swc_path, anisotropy=[1.0, 1.0, 1.0],
):
    """
    """
    file_keys = s3_utils.listdir(bucket, swc_path, s3_client, ext=".swc")
    for node_id, key in enumerate(file_keys):
        # Read and process swc
        print("node_id:", node_id)
        print(key)
        print("")
        raw_swc = s3_utils.read_from_s3(bucket, key, s3_client)
        swc_dict = swc_utils.parse(raw_swc, anisotropy=anisotropy)

        # Feature extraction
        graph.add_node_from_swc(node_id, swc_dict)
        skel_features = extractor.generate_skel_features(swc_dict)
        graph.add_node_feature(skel_features)

        # Store upd_id --> cur_id and xyz --> upd_id
        f = key.split("/")[-1]
        graph.old_node_ids[node_id] = int(f.split(".")[0])
        graph.upd_xyz_to_id(node_id)

    return graph


def create_nodes_from_mask():
    """
    """
    pass


def get_target_labels(bucket, mistake_log_path):
    mistake_log = s3_utils.read_from_s3(bucket, mistake_log_path)
