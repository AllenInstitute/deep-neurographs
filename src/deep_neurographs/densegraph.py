import os

import networkx as nx
import numpy as np
from more_itertools import zip_broadcast
from scipy.spatial import KDTree

from deep_neurographs import swc_utils, utils
from deep_neurographs.geometry_utils import dist, make_line


class DenseGraph:
    def __init__(self, swc_dir):
        self.xyz_to_node = dict()
        self.xyz_to_swc = dict()
        self.init_graphs(swc_dir)
        self.init_kdtree()

    def init_graphs(self, swc_dir):
        self.graphs = dict()
        for f in utils.listdir(swc_dir, ext=".swc"):
            # Extract info
            path = os.path.join(swc_dir, f)

            # Construct Graph
            swc_dict = swc_utils.parse(swc_utils.read_swc(path))
            graph, xyz_to_node = swc_utils.file_to_graph(
                swc_dict, set_attrs=True, return_dict=True
            )

            # Store
            xyz_to_id = dict(zip_broadcast(swc_dict["xyz"], f))
            self.graphs[f] = graph
            self.xyz_to_node[f] = xyz_to_node
            self.xyz_to_swc.update(xyz_to_id)

    def init_kdtree(self):
        self.kdtree = KDTree(list(self.xyz_to_swc.keys()))

    def get_projection(self, xyz):
        _, idx = self.kdtree.query(xyz, k=1)
        proj_xyz = tuple(self.kdtree.data[idx])
        proj_dist = dist(proj_xyz, xyz)
        return proj_xyz, proj_dist

    def connect_nodes(self, graph_id, xyz_i, xyz_j, return_dist=True):
        i = self.xyz_to_node[graph_id][xyz_i]
        j = self.xyz_to_node[graph_id][xyz_j]
        path = nx.shortest_path(self.graphs[graph_id], source=i, target=j)
        if return_dist:
            dist = self.compute_dist(graph_id, path)
            return path, dist
        else:
            return path

    def compute_dist(self, graph_id, path):
        d = 0
        for i in range(1, len(path)):
            xyz_1 = self.graphs[graph_id].nodes[i]["xyz"]
            xyz_2 = self.graphs[graph_id].nodes[i - 1]["xyz"]
            d += dist(xyz_1, xyz_2)
        return d

    def check_aligned(self, pred_xyz_i, pred_xyz_j):
        # Get target graph
        xyz_i, _ = self.get_projection(pred_xyz_i)
        xyz_j, _ = self.get_projection(pred_xyz_j)
        graph_id = self.xyz_to_swc[xyz_i]
        if self.xyz_to_swc[xyz_i] != self.xyz_to_swc[xyz_j]:
            return False

        # Compare pred and target distances
        pred_xyz_i = np.array(pred_xyz_i)
        pred_xyz_j = np.array(pred_xyz_j)
        pred_dist = dist(pred_xyz_i, pred_xyz_j)

        target_path, target_dist = self.connect_nodes(graph_id, xyz_i, xyz_j)
        target_dist = max(target_dist, 1)

        ratio = min(pred_dist, target_dist) / max(pred_dist, target_dist)
        if ratio < 0.6 and pred_dist > 25:
            return False
        elif ratio < 0.25:
            return False

        # Compare projected predicted path
        proj_dists = []
        proj_nodes = set()
        for xyz in make_line(pred_xyz_i, pred_xyz_j, len(target_path)):
            proj_xyz, proj_d = self.get_projection(xyz)
            swc = self.xyz_to_swc[tuple(proj_xyz)]
            proj_nodes.add(self.xyz_to_node[swc][tuple(proj_xyz)])
            proj_dists.append(proj_d)

        intersection = proj_nodes.intersection(set(target_path))
        overlap = len(intersection) / len(target_path)
        if overlap < 0.5 and pred_dist > 25:
            return False
        elif overlap < 0.2:
            return False
        return True
