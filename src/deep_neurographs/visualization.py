"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Subroutines for visualizing neurographs.

"""

import networkx as nx
import numpy as np
import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_connected_components(
    graph, line_width=5, return_data=False, title=""
):
    """
    Visualizes the connected components in "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be visualized.
    line_width : int, optional
        Line width used to plot "subset". The default is 5.
    return_data : bool, optional
        Indication of whether to return data object that is used to generate
        plot. The default is False.
    title : str
        Title of plot. The default is "".

    Returns
    -------
    None or list[graph_objects]

    """
    # Make plot
    data = []
    colors = plc.qualitative.Bold
    connected_components = nx.connected_components(graph)
    cnt = 0
    while True:
        try:
            component = next(connected_components)
            subgraph = graph.subgraph(component)
            color = colors[cnt % len(colors)]
            data.extend(
                plot_edges(
                    graph, subgraph.edges, color=color, line_width=line_width
                )
            )
            cnt += 1
        except StopIteration:
            break

    # Output
    if return_data:
        return data
    else:
        plot(data, title)


def visualize_graph(graph, title=""):
    """
    Visualizes the graph with nodes and edges.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be visualized.
    title : str, optional
        Title of the plot. Default is "".

    Returns
    -------
    None

    """
    data = plot_edges(graph, graph.edges)
    data.append(plot_nodes(graph))
    plot(data, title)


def visualize_proposals(graph, target_graph=None, title="Proposals"):
    """
    Visualizes a graph with proposals.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be visualized.
    target_graph : networkx.Graph, optional
        Graph generated from ground truth tracings. The default is None.
    title : str, optional
        Title of the plot. Default is "Proposals".

    Returns
    -------
    None

    """
    visualize_subset(
        graph,
        graph.proposals,
        proposal_subset=True,
        target_graph=target_graph,
        title=title,
    )


def visualize_targets(
    graph, target_graph=None, title="Ground Truth - Accepted Proposals"
):
    """
    Visualizes a graph and its ground truth accept proposals.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be visualized.
    target_graph : networkx.Graph, optional
        Graph generated from ground truth tracings. The default is None.
    title : str, optional
        Title of the plot. Default is "Ground Truth - Accepted Proposals".

    Returns
    -------
    None

    """
    visualize_subset(
        graph,
        graph.target_edges,
        proposal_subset=True,
        target_graph=target_graph,
        title=title,
    )


def visualize_subset(
    graph,
    subset,
    line_width=5,
    proposal_subset=False,
    target_graph=None,
    title="",
):
    """
    Visualizes a graph and a subset of edges or proposals.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be visualized.
    subset : container
        Subset of edges or proposals to be visualized.
    line_width : int, optional
        Line width used to plot "subset". The default is 5.
    proposals_subset : bool, optional
        Indication of whether "subset" is a subset of proposals. The default
        is False.
    target_graph : networkx.Graph, optional
        Graph generated from ground truth tracings. The default is None.
    title : str, optional
        Title of the plot. Default is "Proposals".

    Returns
    -------
    None

    """
    # Plot graph
    data = plot_edges(graph, graph.edges, color="black")
    data.append(plot_nodes(graph))
    if proposal_subset:
        data.extend(plot_proposals(graph, subset, line_width=line_width))
    else:
        data.extend(plot_edges(graph, subset, line_width=line_width))

    # Add target graph (if applicable)
    if target_graph:
        cc = visualize_connected_components(target_graph, return_data=True)
        data.extend(cc)
    plot(data, title)


# utils
def plot_nodes(graph):
    xyz = nx.get_node_attributes(graph, "xyz")
    xyz = np.array(list(xyz.values()))
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name="Nodes",
        marker=dict(size=2, color="red"),
    )


def plot_proposals(graph, proposals, color=None, line_width=3.5):
    traces = []
    line = (
        dict(width=5) if color is None else dict(color=color, width=line_width)
    )
    for p in proposals:
        xyz = graph.proposal_xyz(p)
        trace = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="lines",
            line=line,
            name="{}".format(tuple(p)),
        )
        traces.append(trace)
    return traces


def plot_edges(graph, edges, color=None, line_width=3.5):
    traces = []
    line = (
        dict(width=5) if color is None else dict(color=color, width=line_width)
    )
    for i, j in edges:
        trace = go.Scatter3d(
            x=graph.edges[i, j]["xyz"][:, 0],
            y=graph.edges[i, j]["xyz"][:, 1],
            z=graph.edges[i, j]["xyz"][:, 2],
            mode="lines",
            line=line,
            name=f"({i},{j})",
        )
        traces.append(trace)
    return traces


def plot(data, title):
    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        template="plotly_white",
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=1)),
        width=1200,
        height=700,
    )
    fig.show()


def subplot(data_1, data_2, title=""):
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]]
    )
    for data in data_1:
        fig.add_trace(data, row=1, col=1)

    for data in data_2:
        fig.add_trace(data, row=1, col=2)

    fig.update_layout(title_text=title, showlegend=True)
    fig.update_xaxes(row=1, col=1, matches="y", showgrid=False)
    fig.update_yaxes(row=1, col=1, matches="x", showgrid=False)
    fig.update_layout(
        scene_aspectmode="manual", scene_aspectratio=dict(x=1, y=1, z=1)
    )

    # Update the size of the second subplot
    fig.update_xaxes(row=1, col=2, matches="y")
    fig.update_yaxes(row=1, col=2, matches="x")
    fig.update_layout(
        scene_aspectmode="manual", scene_aspectratio=dict(x=1, y=1, z=1)
    )

    fig.update_layout(width=1500, height=800)
    fig.show()
