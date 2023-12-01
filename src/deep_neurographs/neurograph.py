"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclass of Networkx.Graph called "NeuroGraph".

"""

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import tensorstore as ts
from copy import deepcopy
from scipy.spatial import KDTree
from time import time

from deep_neurographs import geometry_utils
from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils
from deep_neurographs.densegraph import DenseGraph
from deep_neurographs.geometry_utils import dist as get_dist

BUFFER = 5
SUPPORTED_LABEL_MASK_TYPES = [dict, np.array, ts.TensorStore]


class NeuroGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to irreducible nodes from the
    predicted swc files. This type of graph has two sets of edges referred
    to as "mutable" and "immutable".

    """

    def __init__(self, swc_path, img_path=None, label_mask=None, optimize_proposals=False, origin=None, shape=None):
        """
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        super(NeuroGraph, self).__init__()
        self.path = swc_path
        self.label_mask = label_mask
        self.leafs = set()
        self.junctions = set()

        self.immutable_edges = set()
        self.mutable_edges = set()
        self.target_edges = set()
        self.xyz_to_edge = dict()

        self.img_path = img_path
        self.optimize_proposals = optimize_proposals

        if origin and shape:
            self.bbox = {
                "min": np.array(origin),
                "max": np.array([origin[i] + shape[i] for i in range(3)]),
            }
            self.origin = np.array(origin)
            self.shape = shape
        else:
            self.bbox = None

    # --- Add nodes or edges ---
    def generate_immutables(
        self, swc_id, swc_dict, prune=True, prune_depth=16
    ):
        """
        Adds nodes to graph from a dictionary generated from an swc files.

        Parameters
        ----------
        node_id : int
            Node id.
        swc_dict : dict
            Dictionary generated from an swc where the keys are swc type
            attributes.

        Returns
        -------
        None.

        """
        # Add nodes
        leafs, junctions, edges = gutils.extract_irreducible_graph(
            swc_dict, prune=prune, prune_depth=prune_depth
        )
        node_id = dict()
        for i in leafs + junctions:
            node_id[i] = len(self.nodes)
            self.add_node(
                node_id[i],
                xyz=np.array(swc_dict["xyz"][i]),
                radius=swc_dict["radius"][i],
                swc_id=swc_id,
            )

        # Add edges
        for i, j in edges.keys():
            # Get edge
            edge = (node_id[i], node_id[j])
            xyz = np.array(edges[(i, j)]["xyz"])
            radii = np.array(edges[(i, j)]["radius"])

            # Add edge
            self.immutable_edges.add(frozenset(edge))
            self.add_edge(
                node_id[i], node_id[j], xyz=xyz, radius=radii, swc_id=swc_id
            )
            xyz_to_edge = dict((tuple(xyz), edge) for xyz in xyz)
            check_xyz = set(xyz_to_edge.keys())
            collisions = check_xyz.intersection(set(self.xyz_to_edge.keys()))
            if len(collisions) > 0:
                for xyz in collisions:
                    del xyz_to_edge[xyz]
            self.xyz_to_edge.update(xyz_to_edge)

        # Update leafs and junctions
        for l in leafs:
            self.leafs.add(node_id[l])

        for j in junctions:
            self.junctions.add(node_id[j])

        # Build kdtree
        self._init_kdtree()

    def init_immutable_graph(self):
        immutable_graph = nx.Graph()
        immutable_graph.add_nodes_from(self)
        immutable_graph.add_edges_from(self.immutable_edges)
        return immutable_graph

    def generate_proposals(self, num_proposals=3, search_radius=25.0):
        """
        Generates edges for the graph.

        Returns
        -------
        None

        """
        self.mutable_edges = set()
        for leaf in self.leafs:
            if not self.is_contained(leaf):
                continue
            xyz_leaf = self.nodes[leaf]["xyz"]
            proposals = self._get_proposals(
                leaf, xyz_leaf, num_proposals, search_radius
            )
            for xyz in proposals:
                # Extract info on mutable connection
                (i, j) = self.xyz_to_edge[xyz]
                attrs = self.get_edge_data(i, j)

                # Get connecting node
                contained_j = self.is_contained(j)
                if get_dist(xyz, attrs["xyz"][0]) < 10 and self.is_contained(i):
                    node = i
                    xyz = self.nodes[node]["xyz"]
                elif (
                    get_dist(xyz, attrs["xyz"][-1]) < 10
                    and contained_j
                ):
                    node = j
                    xyz = self.nodes[node]["xyz"]
                else:
                    idxs = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0]
                    node = self.add_immutable_node((i, j), attrs, idxs[0])

                # Add edge
                self.add_edge(leaf, node, xyz=np.array([xyz_leaf, xyz]))
                self.mutable_edges.add(frozenset((leaf, node)))

        if self.optimize_proposals:
            self.run_optimization()


    def _get_proposals(self, query_id, query_xyz, num_proposals, search_radius):
        """
        Parameters
        ----------
        query_id : int
            Node id of the query node.
        query_xyz : tuple[float]
            The (x,y,z) coordinates of the query node.

        Returns
        -------
        None.

        """
        best_xyz = dict()
        best_dist = dict()
        query_swc_id = self.nodes[query_id]["swc_id"]
        for xyz in self._query_kdtree(query_xyz, search_radius):
            if not self.is_contained(xyz):
                continue
            xyz = tuple(xyz)
            edge = self.xyz_to_edge[xyz]
            swc_id = gutils.get_edge_attr(self, edge, "swc_id")
            if swc_id != query_swc_id:
                d = get_dist(xyz, query_xyz)
                if swc_id not in best_dist.keys():
                    best_xyz[swc_id] = xyz
                    best_dist[swc_id] = d
                elif d < best_dist[swc_id]:
                    best_xyz[swc_id] = xyz
                    best_dist[swc_id] = d
        return self._get_best_edges(best_dist, best_xyz, num_proposals)

    def _get_best_edges(self, dists, xyz, num_proposals):
        """
        Gets the at most "num_proposals" nodes that are closest to the
        target node.

        """
        if len(dists.keys()) > num_proposals:
            keys = sorted(dists, key=dists.__getitem__)
            return [xyz[key] for key in keys[0:num_proposals]]
        else:
            return list(xyz.values())

    def add_immutable_node(self, edge, attrs, idx):
        # Remove old edge
        (i, j) = edge
        self.remove_edge(i, j)
        self.immutable_edges.remove(frozenset(edge))

        # Add new node and split edge
        node_id = len(self.nodes)
        self.add_node(
            node_id,
            xyz=tuple(attrs["xyz"][idx]),
            radius=attrs["radius"][idx],
            swc_id=attrs["swc_id"],
        )
        self._add_edge((i, node_id), attrs, np.arange(0, idx + 1))
        self._add_edge((node_id, j), attrs, np.arange(idx, len(attrs["xyz"])))
        return node_id

    def _add_edge(self, edge, attrs, idxs):
        self.add_edge(
            edge[0],
            edge[1],
            xyz=attrs["xyz"][idxs],
            radius=attrs["radius"][idxs],
            swc_id=attrs["swc_id"],
        )
        for xyz in attrs["xyz"][idxs]:
            self.xyz_to_edge[tuple(xyz)] = edge
        self.immutable_edges.add(frozenset(edge))

    def _init_kdtree(self):
        """
        Builds a KD-Tree from the (x,y,z) coordinates of the subnodes of
        each node in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.kdtree = KDTree(list(self.xyz_to_edge.keys()))

    def _query_kdtree(self, query, d):
        """
        Parameters
        ----------
        query : int
            Node id.
        d : float
            Distance from query that is searched.

        Returns
        -------
        generator[tuple]
            Generator that generates the xyz coordinates cooresponding to all
            nodes within a distance of "d" from "query".

        """
        idxs = self.kdtree.query_ball_point(query, d, return_sorted=True)
        return self.kdtree.data[idxs]
    
    def run_optimization(self):
        t0 = time()
        origin = utils.apply_anisotropy(self.origin, return_int=True)
        img = utils.get_superchunk(
            self.img_path, "zarr", origin, self.shape, from_center=False
        )
        img = utils.normalize_img(img)
        simple_edges = self.get_simple_proposals()
        complex_edges = self.get_complex_proposals()
        for edge in self.mutable_edges:
            if edge in simple_edges:
                self.optimize_simple_edge(img, edge)
            else:
                self.optimize_complex_edge(img, edge)
        print("")
        print("edge_optimization(): {} seconds / edge".format((time() - t0) / len(self.get_simple_proposals())))

    def optimize_simple_edge(self, img, edge):
        # Extract Branches
        i, j = tuple(edge)
        xyz_i = self.nodes[i]["xyz"]
        xyz_j = self.nodes[j]["xyz"]
        branch_i = self.get_branch(xyz_i)
        branch_j = self.get_branch(xyz_j)
        
        # Get image patch
        hat_xyz_i = self.to_img(branch_i[8])
        hat_xyz_j = self.to_img(branch_j[8])
        patch_dims = geometry_utils.get_optimal_patch(hat_xyz_i, hat_xyz_j)
        center = geometry_utils.compute_midpoint(hat_xyz_i, hat_xyz_j).astype(int)
        img_chunk = utils.get_chunk(img, center, patch_dims)

        # Optimize
        path = geometry_utils.shortest_path(
            img_chunk,
            utils.img_to_patch(hat_xyz_i, center, patch_dims),
            utils.img_to_patch(hat_xyz_j, center, patch_dims),
        )
        origin = utils.apply_anisotropy(self.origin, return_int=True)
        path = geometry_utils.transform_path(path, origin, center, patch_dims)
        self.edges[edge]["xyz"] = np.vstack([branch_i[8], path, branch_j[8]])

    def get_branch(self, xyz):
        edge = self.xyz_to_edge[tuple(xyz)]
        branch = self.edges[edge]["xyz"]
        if not (branch[0] == xyz).all():
            return np.flip(branch, axis=0)
        else:
            return branch

    def optimize_complex_edge(self, img_superchunk, edge):
        pass

    def init_targets(self, target_neurograph):
        self.target_edges = set()
        self.groundtruth_graph = self.init_immutable_graph()
        target_densegraph = DenseGraph(target_neurograph.path)

        predicted_graph = self.init_immutable_graph()
        site_to_site = dict()
        pair_to_edge = dict()

        proposals = list(self.mutable_edges)
        dists = [self.compute_length(edge) for edge in proposals]
        for idx in np.argsort(dists):
            # Check for cycle
            edge = proposals[idx]
            i, j = tuple(edge)
            if self.check_cycle((i, j)):
                continue

            # Check projection
            xyz_i = self.edges[edge]["xyz"][1]
            xyz_j = self.edges[edge]["xyz"][-1]
            proj_xyz_i, d_i = target_neurograph.get_projection(xyz_i)
            proj_xyz_j, d_j = target_neurograph.get_projection(xyz_j)
            if d_i > 5 or d_j > 5:
                continue

            # Check cases
            edge_i = target_neurograph.xyz_to_edge[proj_xyz_i]
            edge_j = target_neurograph.xyz_to_edge[proj_xyz_j]
            if edge_i != edge_j:
                # Complex criteria
                if not target_neurograph.is_adjacent(edge_i, edge_j):
                    continue
                if not target_densegraph.check_aligned(xyz_i, xyz_j):
                    continue
            else:
                # Simple criteria
                inclusion_i = proj_xyz_i in site_to_site.keys()
                inclusion_j = proj_xyz_j in site_to_site.keys()
                leaf_i = gutils.is_leaf(predicted_graph, i)
                leaf_j = gutils.is_leaf(predicted_graph, j)
                if not leaf_i or not leaf_j:
                    None
                    # continue
                elif inclusion_i or inclusion_j:
                    if inclusion_j:
                        proj_xyz_i = proj_xyz_j
                        proj_xyz_k = site_to_site[proj_xyz_j]
                    else:
                        proj_xyz_k = site_to_site[proj_xyz_i]

                    # Compare edge
                    exists = geometry_utils.compare_edges(
                        proj_xyz_i, proj_xyz_j, proj_xyz_k
                    )
                    if exists:
                        site_to_site, pair_to_edge = self.remove_site(
                            site_to_site, pair_to_edge, proj_xyz_i, proj_xyz_k
                        )

            # Add site
            site_to_site, pair_to_edge = self.add_site(
                site_to_site,
                pair_to_edge,
                proj_xyz_i,
                proj_xyz_j,
                proposals[idx],
            )

        # Print results
        # target_ratio = len(self.target_edges) / len(self.mutable_edges)
        # print("# target edges:", len(self.target_edges))
        # print("% target edges in mutable:", target_ratio)

    def check_simple_criteria(self):
        pass
    
    def check_complex_criteria(self):
        pass

    def add_site(self, site_to_site, pair_to_edge, xyz_i, xyz_j, edge):
        self.target_edges.add(edge)
        site_to_site[xyz_i] = xyz_j
        site_to_site[xyz_j] = xyz_i
        pair_to_edge = self._add_pair_edge(pair_to_edge, xyz_i, xyz_j, edge)
        return site_to_site, pair_to_edge

    def remove_site(self, site_to_site, pair_to_edge, xyz_i, xyz_j):
        del site_to_site[xyz_i]
        del site_to_site[xyz_j]
        pair_to_edge = self._remove_pair_edge(pair_to_edge, xyz_i, xyz_j)
        return site_to_site, pair_to_edge

    def _add_pair_edge(self, pair_to_edge, xyz_i, xyz_j, edge):
        key = frozenset([xyz_i, xyz_j])
        if key not in pair_to_edge.keys():
            pair_to_edge[key] = set([edge])
        else:
            pair_to_edge[key].add(edge)
        return pair_to_edge

    def _remove_pair_edge(self, pair_to_edge, xyz_i, xyz_j):
        key = frozenset([xyz_i, xyz_j])
        edges = list(pair_to_edge[key])
        if len(edges) == 1:
            self.target_edges.remove(edges[0])
            del pair_to_edge[key]
        return pair_to_edge

    # --- Visualization ---
    def visualize_immutables(self, title="Immutable Graph", return_data=False):
        """
        Parameters
        ----------
        node_ids : list[int], optional
            List of node ids to be plotted. The default is None.
        edge_ids : list[tuple], optional
            List of edge ids to be plotted. The default is None.

        Returns
        -------
        None.

        """
        data = self._plot_edges(self.immutable_edges)
        data.append(self._plot_nodes())
        if return_data:
            return data
        else:
            utils.plot(data, title)

    def visualize_mutables(self, title="Mutable Graph", return_data=False):
        data = [self._plot_nodes()]
        data.extend(self._plot_edges(self.immutable_edges, color="black"))
        data.extend(self._plot_edges(self.mutable_edges))
        if return_data:
            return data
        else:
            utils.plot(data, title)

    def visualize_targets(
        self, target_graph=None, title="Target Edges", return_data=False
    ):
        data = [self._plot_nodes()]
        data.extend(self._plot_edges(self.immutable_edges, color="black"))
        data.extend(self._plot_edges(self.target_edges))
        if target_graph is not None:
            data.extend(
                target_graph._plot_edges(
                    target_graph.immutable_edges, color="blue"
                )
            )
        if return_data:
            return data
        else:
            utils.plot(data, title)

    def visualize_subset(self, edges, title=""):
        data = [self._plot_nodes()]
        data.extend(self._plot_edges(self.immutable_edges, color="black"))
        data.extend(self._plot_edges(edges))
        utils.plot(data, title)

    def _plot_nodes(self):
        xyz = nx.get_node_attributes(self, "xyz")
        xyz = np.array(list(xyz.values()))
        points = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            name="Nodes",
            marker=dict(size=3, color="red"),
        )
        return points

    def _plot_edges(self, edges, color=None):
        traces = []
        line = dict(width=4) if color is None else dict(color=color, width=3)
        for i, j in edges:
            trace = go.Scatter3d(
                x=self.edges[(i, j)]["xyz"][:, 0],
                y=self.edges[(i, j)]["xyz"][:, 1],
                z=self.edges[(i, j)]["xyz"][:, 2],
                mode="lines",
                line=line,
                name="({},{})".format(i, j),
            )
            traces.append(trace)
        return traces

    # --- Utils ---
    def num_nodes(self):
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

    def num_edges(self):
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

    def num_immutables(self):
        """
        Computes number of immutable edges in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of immutable edges in the graph.

        """
        return len(self.immutable_edges)

    def num_mutables(self):
        """
        Computes number of mutable edges in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of mutable edges in the graph.

        """
        return len(self.mutable_edges)

    def immutable_degree(self, i):
        degree = 0
        for j in self.neighbors(i):
            if frozenset((i, j)) in self.immutable_edges:
                degree += 1
        return degree

    def compute_length(self, edge, metric="l2"):
        i, j = tuple(edge)
        xyz_1, xyz_2 = self.get_edge_attr("xyz", i, j)
        return get_dist(xyz_1, xyz_2, metric=metric)

    def path_length(self, metric="l2"):
        length = 0
        for edge in self.immutable_edges:
            length += self.compute_length(edge, metric=metric)
        return length

    def get_projection(self, xyz):
        _, idx = self.kdtree.query(xyz, k=1)
        proj_xyz = tuple(self.kdtree.data[idx])
        proj_dist = get_dist(proj_xyz, xyz)
        return proj_xyz, proj_dist

    def is_adjacent(self, edge_i, edge_j):
        i, j = tuple(edge_i)
        k, l = tuple(edge_j)
        nb_bool_i = self.is_nb(i, k) or self.is_nb(i, l)
        nb_bool_j = self.is_nb(j, k) or self.is_nb(j, l)
        if nb_bool_i or nb_bool_j:
            return True
        else:
            return False

    def is_nb(self, i, j):
        return True if i in self.neighbors(j) else False

    def is_contained(self, node_or_xyz):
        if self.bbox:
            if type(node_or_xyz) == int:
                node_or_xyz = deepcopy(self.nodes[node_or_xyz]["xyz"])
            xyz = utils.apply_anisotropy(node_or_xyz - self.bbox["min"])
            img_shape = np.array(self.shape)
            for i in range(3):
                lower_bool = xyz[i] < 32
                upper_bool = xyz[i] > img_shape[i] - 32
                if lower_bool or upper_bool:
                    return False
        return True

    def is_leaf(self, i):
        return True if len(self.neighbors(i)) == 1 else False

    def check_cycle(self, edge):
        self.groundtruth_graph.add_edges_from([edge])
        try:
            nx.find_cycle(self.groundtruth_graph)
        except:
            return False
        self.groundtruth_graph.remove_edges_from([edge])
        return True

    def get_edge_attr(self, key, i, j):
        attr_1 = self.nodes[i][key]
        attr_2 = self.nodes[j][key]
        return attr_1, attr_2

    def get_center(self):
        return geometry_utils.compute_midpoint(
            self.bbox["min"], self.bbox["max"]
        )

    def get_complex_proposals(self):
        return set([e for e in self.mutable_edges if not self.is_simple(e)])

    def get_simple_proposals(self):
        return set([e for e in self.mutable_edges if self.is_simple(e)])

    def is_simple(self, edge):
        i, j = tuple(edge)
        if self.immutable_degree(i) == 1 and self.immutable_degree(j) == 1:
            return True
        else:
            return False

    def to_img(self, node_or_xyz):
        if type(node_or_xyz) == int:
            node_or_xyz = deepcopy(self.nodes[node_or_xyz]["xyz"])
        return utils.to_img(node_or_xyz, shift=self.origin)

    def to_line_graph(self):
        """
        Converts graph to a line graph.

        Parameters
        ----------
        None

        Returns
        -------
        networkx.Graph
            Line graph.

        """
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        return nx.line_graph(graph)
