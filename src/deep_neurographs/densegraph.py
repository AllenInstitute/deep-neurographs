"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Class of graphs built from swc files. Each swc file is stored as a distinct
graph and each node in this graph.

"""

import os
from random import sample
from time import time

import networkx as nx

from deep_neurographs import swc_utils, utils


class DenseGraph:
    """
    Class of graphs built from swc files. Each swc file is stored as a
    distinct graph and each node in this graph.

    """

    def __init__(self, swc_paths, image_patch_origin, image_patch_shape):
        """
        Constructs a DenseGraph object from a directory of swc files.

        Parameters
        ----------
        swc_paths : list[str]
            List of paths to swc files which are used to construct a hash
            table in which the entries are filename-graph pairs.

        Returns
        -------
        None

        """
        self.init_graph(swc_paths)
        self.origin = image_patch_origin
        self.shape = image_patch_shape

    def init_graph(self, paths):
        """
        Initializes graphs by reading swc files in "swc_paths". Graphs are
        stored in a hash table where the entries are filename-graph pairs.

        Parameters
        ----------
        swc_paths : list[str]
            List of paths to swc files which are used to construct a hash
            table in which the entries are filename-graph pairs.

        Returns
        -------
        None

        """
        # Initializations
        print("Building graph...")
        self.graph = nx.Graph()
        swc_dicts, _ = swc_utils.process_local_paths(paths)

        # Run
        cnt = 1
        t0, t1 = utils.init_timers()
        chunk_size = max(int(len(swc_dicts) * 0.02), 1)
        for i, swc_dict in enumerate(swc_dicts):
            # Construct Graph
            swc_id = swc_dict["swc_id"]
            graph, _ = swc_utils.to_graph(swc_dict, set_attrs=True)
            graph = add_swc_id(graph, swc_id)
            self.graph = nx.disjoint_union(self.graph, graph)

            # Report progress
            if i > cnt * chunk_size:
                cnt, t1 = report_progress(
                    i, len(swc_dicts), chunk_size, cnt, t0, t1
                )

    def trim(self):
        pass

    def save(self, path):
        for i, component in enumerate(nx.connected_components(self.graph)):
            node = sample(component, 1)[0]
            swc_id = self.graph.nodes[node]["swc_id"]
            component_path = os.path.join(path, f"{swc_id}.swc")
            self.component_to_swc(component_path, component)

    def component_to_swc(self, path, component):
        node_to_idx = dict()
        entry_list = []
        for i, j in nx.dfs_edges(self.graph.subgraph(component)):
            # Initialize
            if len(entry_list) == 0:
                x, y, z = tuple(self.graph.nodes[i]["xyz"])
                r = self.graph.nodes[i]["radius"]
                entry_list.append([1, 2, x, y, z, r, -1])
                node_to_idx[i] = 1

            # Create entry
            node_to_idx[j] = len(entry_list) + 1
            x, y, z = tuple(self.graph.nodes[j]["xyz"])
            r = self.graph.nodes[j]["radius"]
            entry_list.append([node_to_idx[j], 2, x, y, z, r, node_to_idx[i]])

        swc_utils.write(path, entry_list)


def add_swc_id(graph, swc_id):
    for i in graph.nodes:
        graph.nodes[i]["swc_id"] = swc_id
    return graph


def report_progress(current, total, chunk_size, cnt, t0, t1):
    eta = get_eta(current, total, chunk_size, t1)
    runtime = get_runtime(current, total, chunk_size, t0, t1)
    utils.progress_bar(current, total, eta=eta, runtime=runtime)
    return cnt + 1, time()


def get_eta(current, total, chunk_size, t0, return_str=True):
    chunk_runtime = time() - t0
    remaining = total - current
    eta = remaining * (chunk_runtime / chunk_size)
    t, unit = utils.time_writer(eta)
    return f"{round(t, 4)} {unit}" if return_str else eta


def get_runtime(current, total, chunk_size, t0, t1):
    eta = get_eta(current, total, chunk_size, t1, return_str=False)
    total_runtime = time() - t0 + eta
    t, unit = utils.time_writer(total_runtime)
    return f"{round(t, 4)} {unit}"
