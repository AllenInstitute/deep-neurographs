"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclass of Networkx.Graph called "NeuroGraph".

"""
import os
from random import sample

import networkx as nx
import numpy as np
import tensorstore as ts
from scipy.spatial import KDTree

from deep_neurographs import geometry
from deep_neurographs import graph_utils as gutils
from deep_neurographs import swc_utils, utils
from deep_neurographs.geometry import check_dists
from deep_neurographs.geometry import dist as get_dist
from deep_neurographs.groundtruth_generation import init_targets

SUPPORTED_LABEL_MASK_TYPES = [dict, np.array, ts.TensorStore]


class NeuroGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to irreducible nodes from the
    predicted swc files.

    """

    def __init__(
        self,
        img_bbox=None,
        swc_paths=None,
        img_path=None,
        label_mask=None,
        train_model=False,
    ):
        super(NeuroGraph, self).__init__()
        # Initialize paths
        self.img_path = img_path
        self.label_mask = label_mask
        self.swc_paths = swc_paths
        self.swc_ids = set()

        # Initialize node and edge sets
        self.leafs = set()
        self.junctions = set()
        self.proposals = dict()
        self.target_edges = set()

        # Initialize data structures for proposals
        self.complex_proposals = set()
        self.simple_proposals = set()
        self.xyz_to_edge = dict()
        self.kdtree = None

        # Initialize bounding box (if exists)
        self.bbox = img_bbox
        if self.bbox:
            self.origin = img_bbox["min"].astype(int)
            self.shape = (img_bbox["max"] - img_bbox["min"]).astype(int)
        else:
            self.origin = np.array([0, 0, 0], dtype=int)
            self.shape = None

    def copy_graph(self, add_attrs=False):
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes(data=add_attrs))
        if add_attrs:
            for edge in self.edges:
                i, j = tuple(edge)
                graph.add_edge(i, j, **self.get_edge_data(i, j))
        else:
            graph.add_edges_from(self.edges)
        return graph

    # --- Add nodes or edges ---
    def add_swc_id(self, swc_id):
        self.swc_ids.add(swc_id)

    def add_component(self, component_dict):
        # Nodes
        ids = dict()
        cur_id = len(self.nodes) + 1
        ids, cur_id = self.__add_nodes(component_dict, "leafs", ids, cur_id)
        ids, cur_id = self.__add_nodes(
            component_dict, "junctions", ids, cur_id
        )
        self.add_swc_id(component_dict["swc_id"])

        # Add edges
        for edge, attrs in component_dict["edges"].items():
            i, j = edge
            self.add_edge(
                ids[i],
                ids[j],
                radius=attrs["radius"],
                xyz=attrs["xyz"],
                swc_id=component_dict["swc_id"],
            )
            for xyz in attrs["xyz"][::2]:
                self.xyz_to_edge[tuple(xyz)] = (ids[i], ids[j])
            self.xyz_to_edge[tuple(attrs["xyz"][-1])] = (ids[i], ids[j])

    def __add_nodes(self, nodes, key, node_ids, cur_id):
        for i in nodes[key].keys():
            node_ids[i] = cur_id
            self.add_node(
                node_ids[i],
                proposals=set(),
                radius=nodes[key][i]["radius"],
                swc_id=nodes["swc_id"],
                xyz=nodes[key][i]["xyz"],
            )
            if key == "leafs":
                self.leafs.add(cur_id)
            else:
                self.junctions.add(cur_id)
            cur_id += 1
        return node_ids, cur_id

    def __add_edge(self, edge, attrs, idxs):
        self.add_edge(
            edge[0],
            edge[1],
            xyz=attrs["xyz"][idxs],
            radius=attrs["radius"][idxs],
            swc_id=attrs["swc_id"],
        )
        for xyz in attrs["xyz"][idxs]:
            self.xyz_to_edge[tuple(xyz)] = edge

    # --- Proposal and Ground Truth Generation ---
    def generate_proposals(
        self,
        radius,
        filter_doubles=False,
        proposals_per_leaf=3,
        optimize=False,
        optimization_depth=10,
        restrict=True,
    ):
        """
        Generates edges for the graph.

        Returns
        -------
        None

        """
        self.init_kdtree()
        self.reset_proposals()
        self.proposals = dict()
        existing_connections = dict()
        doubles = set()
        not_doubles = set()
        for leaf in self.leafs:
            # Check if leaf is contained in bbox (if applicable)
            if not self.is_contained(leaf, buffer=36):
                continue

            # Check if leaf is double (if applicable)
            leaf_swc_id = self.nodes[leaf]["swc_id"]
            doubles, not_doubles = self.upd_doubles(
                leaf, doubles, not_doubles, filter_doubles
            )
            if leaf_swc_id in doubles:
                continue

            # Check proposals
            xyz_leaf = self.nodes[leaf]["xyz"]
            self.set_proposals_per_leaf(proposals_per_leaf)
            for xyz in self.__get_proposals(leaf, xyz_leaf, radius):
                # Extract info on proposal
                (i, j) = self.xyz_to_edge[xyz]
                attrs = self.get_edge_data(i, j)
                swc_id = self.nodes[i]["swc_id"]

                # Check if double
                doubles, not_doubles = self.upd_doubles(
                    i, doubles, not_doubles, filter_doubles
                )
                if swc_id in doubles:
                    continue

                # Check connection
                node, xyz = self.__get_connecting_node(
                    xyz_leaf, xyz, (i, j), attrs, radius
                )
                if self.degree[node] >= 3:
                    continue

                pair_id = frozenset((swc_id, leaf_swc_id))
                if pair_id in existing_connections.keys() and restrict:
                    edge = existing_connections[pair_id]
                    len1 = self.node_xyz_dist(leaf, xyz)
                    len2 = self.proposal_length(edge)
                    if len1 < len2:
                        node1, node2 = tuple(edge)
                        self.nodes[node1]["proposals"].discard(node2)
                        self.nodes[node2]["proposals"].discard(node1)
                        del self.proposals[edge]
                        del existing_connections[pair_id]
                    else:
                        continue

                # Add proposal
                if self.degree[node] < 2:
                    edge = frozenset({leaf, node})
                    self.nodes[node]["proposals"].add(leaf)
                    self.nodes[leaf]["proposals"].add(node)
                    self.proposals[edge] = {"xyz": np.array([xyz_leaf, xyz])}
                    existing_connections[pair_id] = edge

        # print("# doubles:", len(doubles))

        # Finish
        self.filter_nodes()
        if optimize:
            self.run_optimization()

    def reset_proposals(self):
        for i in self.nodes:
            self.nodes[i]["proposals"] = set()

    def set_proposals_per_leaf(self, proposals_per_leaf):
        self.proposals_per_leaf = proposals_per_leaf

    def __get_proposals(self, query_id, query_xyz, radius):
        """
        Generates edge proposals for node "query_id" by finding points on
        distinct connected components near "query_xyz".

        Parameters
        ----------
        query_id : int
            Node id of the query node.
        query_xyz : tuple[float]
            (x,y,z) coordinates of the query node.
        radius : float
            Maximum Euclidean length of edge proposal.

        Returns
        -------
        list
            List of best edge proposals generated from "query_node".

        """
        best_xyz = dict()
        best_dist = dict()
        query_swc_id = self.nodes[query_id]["swc_id"]
        for xyz in self.query_kdtree(query_xyz, radius):
            # Check whether xyz is contained
            if not self.is_contained(xyz, buffer=36):
                continue

            # Check whether
            edge = self.xyz_to_edge[tuple(xyz)]
            swc_id = gutils.get_edge_attr(self, edge, "swc_id")
            if swc_id != query_swc_id:
                d = get_dist(xyz, query_xyz)
                if swc_id not in best_dist.keys():
                    best_xyz[swc_id] = tuple(xyz)
                    best_dist[swc_id] = d
                elif d < best_dist[swc_id]:
                    best_xyz[swc_id] = tuple(xyz)
                    best_dist[swc_id] = d
        return self.get_best_proposals(best_dist, best_xyz)

    def get_best_proposals(self, dists, xyz):
        """
        Gets the at most "self.proposals_per_leaf" nodes that are closest to
        "xyz".

        """
        if len(dists.keys()) > self.proposals_per_leaf:
            keys = sorted(dists, key=dists.__getitem__)
            return [xyz[key] for key in keys[0 : self.proposals_per_leaf]]
        else:
            return list(xyz.values())

    def split_edge(self, edge, attrs, idx):
        """
        Splits "edge" into two distinct edges by making the subnode at "idx" a
        new node in "self".

        Parameters
        ----------
        edge : tuple
            Edge to be split.
        attrs : dict
            Attributes of "edge".
        idx : int
            Index of subnode that will become a new node in "self".

        Returns
        -------
        new_node : int
            Node ID of node that was created.

        """
        # Remove old edge
        (i, j) = edge
        self.remove_edge(i, j)

        # Add new node and split edge
        new_node = len(self.nodes) + 1
        self.add_node(
            new_node,
            proposals=set(),
            radius=attrs["radius"][idx],
            swc_id=attrs["swc_id"],
            xyz=tuple(attrs["xyz"][idx]),
        )
        self.__add_edge((i, new_node), attrs, np.arange(0, idx + 1))
        self.__add_edge(
            (new_node, j), attrs, np.arange(idx, len(attrs["xyz"]))
        )
        return new_node

    def __get_connecting_node(self, xyz_leaf, xyz_edge, edge, attrs, radius):
        i, j = tuple(edge)
        d_i = check_dists(xyz_leaf, xyz_edge, self.nodes[i]["xyz"], radius)
        d_j = check_dists(xyz_leaf, xyz_edge, self.nodes[j]["xyz"], radius)
        if d_i and self.is_contained(i, buffer=36):
            return i, self.nodes[i]["xyz"]
        elif d_j and self.is_contained(j, buffer=36):
            return j, self.nodes[j]["xyz"]
        else:
            idxs = np.where(np.all(attrs["xyz"] == xyz_edge, axis=1))[0]
            node = self.split_edge((i, j), attrs, idxs[0])
            return node, xyz_edge

    def init_targets(self, target_neurograph):
        target_neurograph.init_kdtree()
        self.target_edges = init_targets(target_neurograph, self)

    def run_optimization(self):
        driver = "n5" if ".n5" in self.img_path else "zarr"
        img = utils.get_superchunk(
            self.img_path, driver, self.origin, self.shape, from_center=False
        )
        for edge in self.proposals:
            xyz_1, xyz_2 = geometry.optimize_alignment(self, img, edge)
            self.proposals[edge]["xyz"] = np.array([xyz_1, xyz_2])

    # -- kdtree --
    def init_kdtree(self):
        """
        Builds a KD-Tree from the (x,y,z) coordinates of the subnodes of
        each connected component in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if not self.kdtree:
            self.kdtree = KDTree(list(self.xyz_to_edge.keys()))

    def query_kdtree(self, xyz, d):
        """
        Parameters
        ----------
        xyz : int
            Node id.
        d : float
            Distance from "xyz" that is searched.

        Returns
        -------
        generator[tuple]
            Generator that generates the xyz coordinates cooresponding to all
            nodes within a distance of "d" from "xyz".

        """
        idxs = self.kdtree.query_ball_point(xyz, d, return_sorted=True)
        return self.kdtree.data[idxs]

    def get_projection(self, xyz):
        _, idx = self.kdtree.query(xyz, k=1)
        return tuple(self.kdtree.data[idx])

    # --- utils ---
    def n_nodes(self):
        """
        Computes number of nodes in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of nodes in the graph.

        """
        return self.number_of_nodes()

    def n_edges(self):
        """
        Computes number of edges in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of edges in the graph.

        """
        return self.number_of_edges()

    def n_proposals(self):
        """
        Computes number of edges proposals in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of edge proposals in the graph.

        """
        return len(self.proposals)

    def get_proposals(self):
        return list(self.proposals.keys())

    def remove_proposal(self, edge):
        del self.proposals[edge]

    def proposal_xyz(self, edge):
        return tuple(self.proposals[edge]["xyz"])

    def proposal_length(self, edge):
        i, j = tuple(edge)
        return get_dist(self.nodes[i]["xyz"], self.nodes[j]["xyz"])

    def get_branches(self, i, key="xyz"):
        branches = []
        for j in self.neighbors(i):
            branches.append(self.orient_edge((i, j), i, key=key))
        return branches

    def orient_edge(self, edge, i, key="xyz"):
        if (self.edges[edge][key][0] == self.nodes[i][key]).all():
            return self.edges[edge][key]
        else:
            return np.flip(self.edges[edge][key], axis=0)

    def node_xyz_dist(self, node, xyz):
        return get_dist(xyz, self.nodes[node]["xyz"])

    def is_nb(self, i, j):
        return True if i in self.neighbors(j) else False

    def is_contained(self, node_or_xyz, buffer=0):
        if self.bbox:
            img_coord = self.to_img(node_or_xyz)
            return utils.is_contained(self.bbox, img_coord, buffer=buffer)
        else:
            return True

    def branch_contained(self, xyz_list):
        if self.bbox:
            return all(
                [self.is_contained(xyz, buffer=-32) for xyz in xyz_list]
            )
        else:
            return True

    def to_img(self, node_or_xyz, shift=False):
        shift = self.origin if shift else np.zeros((3))
        if type(node_or_xyz) == int:
            img_coord = utils.to_img(self.nodes[node_or_xyz]["xyz"])
        else:
            img_coord = utils.to_img(node_or_xyz)
        return img_coord - shift

    def is_leaf(self, i):
        return True if self.degree[i] == 1 else False

    def get_edge_attr(self, edge, key):
        xyz_arr = gutils.get_edge_attr(self, edge, key)
        return xyz_arr[0], xyz_arr[-1]

    def get_complex_proposals(self):
        return set([e for e in self.get_proposals() if not self.is_simple(e)])

    def get_simple_proposals(self):
        return set([e for e in self.get_proposals() if self.is_simple(e)])

    def is_simple(self, edge):
        i, j = tuple(edge)
        return True if self.is_leaf(i) and self.is_leaf(j) else False

    def to_patch_coords(self, edge, midpoint, chunk_size):
        patch_coords = []
        for xyz in self.edges[edge]["xyz"]:
            img_coord = self.to_img(xyz)
            coord = utils.img_to_patch(img_coord, midpoint, chunk_size)
            patch_coords.append(coord)
        return patch_coords

    def get_reconstruction(self, proposals, upd_self=False):
        reconstruction = self.copy_graph(add_attrs=True)
        for edge in proposals:
            i, j = tuple(edge)
            r_i = self.nodes[i]["radius"]
            r_j = self.nodes[j]["radius"]
            reconstruction.add_edge(
                i, j, xyz=self.proposals[i, j]["xyz"], radius=[r_i, r_j]
            )
        return reconstruction

    def upd_doubles(self, i, doubles, not_doubles, filter_doubles):
        if filter_doubles:
            swc_id_i = self.nodes[i]["swc_id"]
            if swc_id_i not in doubles.union(not_doubles):
                if self.is_double(i):
                    doubles.add(swc_id_i)
                else:
                    not_doubles.add(swc_id_i)
        return doubles, not_doubles

    def is_double(self, i):
        """
        Determines whether the connected component corresponding to "root" is
        a double of another connected component.

        Paramters
        ---------
        root : int
            Node of connected component to be evaluated.

        Returns
        -------
        bool
            Indication of whether connected component is a double.

        """
        nb = list(self.neighbors(i))[0]
        if self.degree[i] == 1 and self.degree[nb] == 1:
            # Find near components
            swc_id_i = self.nodes[i]["swc_id"]
            hits = dict()  # near components
            segment_i = self.get_branches(i)[0]
            for xyz_i in segment_i:
                for xyz_j in self.query_kdtree(xyz_i, 8):
                    swc_id_j, node = self.xyz_to_swc(xyz_j, return_node=True)
                    if swc_id_i != swc_id_j:
                        hits = utils.append_dict_value(hits, swc_id_j, node)
                        break

            # Parse queried components
            swc_id_j, n_close = utils.find_best(hits)
            percent_close = n_close / len(segment_i)
            if swc_id_j is not None and percent_close > 0.5:
                j = sample(hits[swc_id_j], 1)[0]
                length_i = len(segment_i)
                length_j = self.component_cardinality(j)
                if length_i / length_j < 0.6:
                    # print("swc_id_i:", swc_id_i)
                    # print("swc_id_j:", swc_id_j)
                    # print("% branch hit:", percent_close)
                    # print("length ratio:", length_i / length_j)
                    # print("double:", swc_id_i)
                    # print("")
                    return True
        return False

    def xyz_to_swc(self, xyz, return_node=False):
        edge = self.xyz_to_edge[tuple(xyz)]
        i, j = tuple(edge)
        if return_node:
            return self.edges[edge]["swc_id"], i
        else:
            return self.edges[edge]["swc_id"]

    def component_cardinality(self, root):
        cardinality = 0
        queue = [(-1, root)]
        visited = set()
        while len(queue):
            # Visit
            i, j = queue.pop()
            visited.add(frozenset((i, j)))
            if i != -1:
                cardinality = len(self.edges[i, j]["xyz"])

            # Add neighbors
            for k in self.neighbors(j):
                if frozenset((j, k)) not in visited:
                    queue.append((j, k))
        return cardinality

    def filter_nodes(self):
        # Find nodes to filter
        ingest_nodes = set()
        for i in [i for i in self.nodes if self.degree[i] == 2]:
            if len(self.nodes[i]["proposals"]) == 0:
                ingest_nodes.add(i)

        # Ingest nodes to be filtered
        for i in ingest_nodes:
            nbs = list(self.neighbors(i))
            self.absorb_node(i, nbs[0], nbs[1])

    def absorb_node(self, i, nb_1, nb_2):
        # Get attributes
        xyz = self.get_branches(i, key="xyz")
        radius = self.get_branches(i, key="radius")

        # Edit graph
        self.remove_node(i)
        self.add_edge(
            nb_1,
            nb_2,
            xyz=np.vstack([np.flip(xyz[1], axis=0), xyz[0][1:, :]]),
            radius=np.concatenate((radius[0], np.flip(radius[1]))),
            swc_id=self.nodes[nb_1]["swc_id"],
        )

    def merge_proposal(self, edge):
        # Attributes
        i, j = tuple(edge)
        xyz = np.vstack([self.nodes[i]["xyz"], self.nodes[j]["xyz"]])
        radius = np.array([self.nodes[i]["radius"], self.nodes[j]["radius"]])

        # Add
        self.add_edge(i, j, xyz=xyz, radius=radius, swc_id="merged")
        del self.proposals[edge]

    def to_swc(self, path):
        for i, component in enumerate(nx.connected_components(self)):
            node = sample(component, 1)[0]
            swc_id = self.nodes[node]["swc_id"]
            component_path = os.path.join(path, f"{swc_id}.swc")
            self.component_to_swc(component_path, component)

    def component_to_swc(self, path, component):
        node_to_idx = dict()
        entry_list = []
        for i, j in nx.dfs_edges(self.subgraph(component)):
            # Initialize
            if len(entry_list) == 0:
                x, y, z = tuple(self.nodes[i]["xyz"])
                r = self.nodes[i]["radius"]
                entry_list.append([1, 2, x, y, z, r, -1])
                node_to_idx[i] = 1

            # Create entry
            parent = node_to_idx[i]
            entry_list = self.branch_to_entries(entry_list, i, j, parent)
            node_to_idx[j] = len(entry_list)
        swc_utils.write(path, entry_list)

    def branch_to_entries(self, entry_list, i, j, parent):
        # Orient branch
        branch_xyz = self.edges[i, j]["xyz"]
        branch_radius = self.edges[i, j]["radius"]
        if (branch_xyz[0] != self.nodes[i]["xyz"]).any():
            branch_xyz = np.flip(branch_xyz, axis=0)
            branch_radius = np.flip(branch_radius, axis=0)

        # Make entries
        for k in range(1, len(branch_xyz)):
            x, y, z = tuple(branch_xyz[k])
            r = branch_radius[k]
            node_id = len(entry_list) + 1
            parent = len(entry_list) if k > 1 else parent
            entry_list.append([node_id, 2, x, y, z, r, parent])

        return entry_list
