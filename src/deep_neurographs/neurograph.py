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
from scipy.spatial import KDTree

from deep_neurographs import geometry_utils
from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils
from deep_neurographs.densegraph import DenseGraph

SUPPORTED_LABEL_MASK_TYPES = [dict, np.array, ts.TensorStore]


class NeuroGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to irreducible nodes from the
    predicted swc files. This type of graph has two sets of edges referred
    to as "mutable" and "immutable".

    """

    def __init__(self, swc_path, label_mask=None, origin=None, shape=None):
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
        if origin is not None and shape is not None:
            self.init_bbox(origin, shape)
        else:
            self.bbox = None

    # --- Add nodes or edges ---
    def init_bbox(self, origin, shape):
        self.bbox = dict()
        self.bbox["min"] = list(origin)
        self.bbox["max"] = [self.bbox["min"][i] + shape[i] for i in range(3)]

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

    def init_immutable_graph(self):
        immutable_graph = nx.Graph()
        immutable_graph.add_nodes_from(self)
        immutable_graph.add_edges_from(self.immutable_edges)
        return immutable_graph

    def generate_mutables(self, max_degree=3, search_radius=25.0):
        """
        Generates edges for the graph.

        Returns
        -------
        None

        """
        # Search for mutable connections
        self.mutable_edges = set()
        self._init_kdtree()
        for leaf in self.leafs:
            xyz_leaf = self.nodes[leaf]["xyz"]
            if not self.is_contained(xyz_leaf):
                continue
            mutables = self._get_mutables(
                leaf, xyz_leaf, max_degree, search_radius
            )
            for xyz in mutables:
                if not self.is_contained(xyz):
                    continue
                # Extract info on mutable connection
                (i, j) = self.xyz_to_edge[xyz]
                attrs = self.get_edge_data(i, j)

                # Get connecting node
                if geometry_utils.dist(xyz, attrs["xyz"][0]) < 10:
                    node = i
                    xyz = self.nodes[node]["xyz"]
                elif geometry_utils.dist(xyz, attrs["xyz"][-1]) < 10:
                    node = j
                    xyz = self.nodes[node]["xyz"]
                else:
                    idxs = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0]
                    node = self.add_immutable_node((i, j), attrs, idxs[0])

                # Add edge
                self.add_edge(leaf, node, xyz=np.array([xyz_leaf, xyz]))
                self.mutable_edges.add(frozenset((leaf, node)))

    def _get_mutables(self, query_id, query_xyz, max_degree, search_radius):
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
            xyz = tuple(xyz)
            edge = self.xyz_to_edge[xyz]
            swc_id = gutils.get_edge_attr(self, edge, "swc_id")
            if swc_id != query_swc_id:
                d = geometry_utils.dist(xyz, query_xyz)
                if swc_id not in best_dist.keys():
                    best_xyz[swc_id] = xyz
                    best_dist[swc_id] = d
                elif d < best_dist[swc_id]:
                    best_xyz[swc_id] = xyz
                    best_dist[swc_id] = d
        return self._get_best_edges(best_dist, best_xyz, max_degree)

    def _get_best_edges(self, dist, xyz, max_degree):
        """
        Gets the at most "max_degree" nodes that are closest to the
        target node.

        Parameters
        ----------
        best_dist : dict
            Dictionary where the keys are node_ids and values are distance
            from the target node.

        Returns
        -------
        best_dist : dict
            Dictionary of nodes that are closest to the target node.

        """
        if len(dist.keys()) > max_degree:
            keys = sorted(dist, key=dist.__getitem__)
            return [xyz[key] for key in keys[0:max_degree]]
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

    def _query_kdtree(self, query, dist):
        """
        Parameters
        ----------
        query : int
            Node id.
        dist : float
            Distance from query that is searched.

        Returns
        -------
        generator[tuple]
            Generator that generates the xyz coordinates cooresponding to all
            nodes within a distance of "dist" from "query".

        """
        idxs = self.kdtree.query_ball_point(query, dist, return_sorted=True)
        return self.kdtree.data[idxs]

    def init_targets(self, target_neurograph):
        self.target_edges = set()
        self.groundtruth_graph = self.init_immutable_graph()
        target_densegraph = DenseGraph(target_neurograph.path)

        predicted_graph = self.init_immutable_graph()
        site_to_site = dict()
        pair_to_edge = dict()

        mutable_edges = list(self.mutable_edges)
        dists = [self.compute_length(edge) for edge in mutable_edges]
        for idx in np.argsort(dists):
            # Get projection
            i, j = tuple(mutable_edges[idx])
            xyz_i = self.nodes[i]["xyz"]
            xyz_j = self.nodes[j]["xyz"]
            proj_xyz_i, d_i = target_neurograph.get_projection(xyz_i)
            proj_xyz_j, d_j = target_neurograph.get_projection(xyz_j)

            # Check criteria
            if d_i > 8 or d_j > 8:
                continue
            elif self.check_cycle((i, j)):
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
                mutable_edges[idx],
            )

        # Print results
        # target_ratio = len(self.target_edges) / len(self.mutable_edges)
        # print("# target edges:", len(self.target_edges))
        # print("% target edges in mutable:", target_ratio)

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
        return geometry_utils.dist(xyz_1, xyz_2, metric=metric)

    def path_length(self, metric="l2"):
        length = 0
        for edge in self.immutable_edges:
            length += self.compute_length(edge, metric=metric)
        return length

    def get_projection(self, xyz):
        _, idx = self.kdtree.query(xyz, k=1)
        proj_xyz = tuple(self.kdtree.data[idx])
        proj_dist = geometry_utils.dist(proj_xyz, xyz)
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

    def is_contained(self, xyz):
        if type(self.bbox) is dict:
            for i in range(3):
                lower_bool = xyz[i] < self.bbox["min"][i]
                upper_bool = xyz[i] > self.bbox["max"][i]
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


# Check whether to trim end of branch
"""
if node_id[i] in leafs:
    ref_xyz = copy.deepcopy(self.nodes[node_id[i]]["xyz"])
    xyz, radii, idx = geometry_utils.smooth_end(xyz, radii, ref_xyz, num_pts=3)
    if idx is not None:
    nx.set_node_attributes(self, {node_id[i]: xyz[idx, :]}, 'xyz')
    nx.set_node_attributes(self, {node_id[i]: radii[idx]}, 'radius')

if node_id[j] in leafs:
    ref_xyz = copy.deepcopy(self.nodes[node_id[i]]["xyz"])
    xyz, radii, idx = geometry_utils.smooth_end(xyz, radii, ref_xyz, num_pts=3)
    if idx is not None:
        self.nodes[node_id[j]]["xyz"] = xyz[idx]
        self.nodes[node_id[j]]["radius"] = radii[idx]
"""
