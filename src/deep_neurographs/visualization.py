"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for visualizing FragmentsGraph.

"""

from plotly.subplots import make_subplots

import networkx as nx
import plotly.colors as plc
import plotly.graph_objects as go


def visualize_connected_components(
    graph, width=3, return_data=False, title=""
):
    """
    Visualizes the connected components in "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be visualized.
    width : int, optional
        Line width used to plot edges in "subset". The default is 5.
    return_data : bool, optional
        Indication of whether to return data object that is used to generate
        plot. The default is False.
    title : str
        Title of plot. The default is an empty string.

    Returns
    -------
    None or List[graph_objects]
    """
    # Initializations
    colors = plc.qualitative.Bold
    connected_components = nx.connected_components(graph)

    # Generate plot data
    data = []
    while True:
        try:
            component = next(connected_components)
            subgraph = graph.subgraph(component)
            color = colors[len(data) % len(colors)]
            data.extend(
                get_edge_traces(
                    graph, subgraph.edges, color=color, width=width
                )
            )
        except StopIteration:
            break

    # Finish
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
    """
    data = get_edge_traces(graph, graph.edges)
    data.append(get_node_traces(graph))
    plot(data, title)


def visualize_proposals(
    graph, color=None, groundtruth_graph=None, title="Proposals"
):
    """
    Visualizes a graph and its proposals.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be visualized.
    groundtruth_graph : networkx.Graph, optional
        Graph generated from groundtruth tracings. The default is None.
    title : str, optional
        Title of the plot. Default is "Proposals".
    """
    visualize_subset(
        graph,
        graph.proposals,
        color=color,
        is_proposal_subset=True,
        groundtruth_graph=groundtruth_graph,
        title=title,
    )


def visualize_groundtruth(
    graph, groundtruth_graph=None, title="Ground Truth - Accepted Proposals"
):
    """
    Visualizes a graph and its groundtruth accepted proposals.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be visualized.
    groundtruth_graph : networkx.Graph, optional
        Graph generated from groundtruth tracings. The default is None.
    title : str, optional
        Title of the plot. Default is "Ground Truth - Accepted Proposals".
    """
    visualize_subset(
        graph,
        graph.target_edges,
        is_proposal_subset=True,
        groundtruth_graph=groundtruth_graph,
        title=title,
    )


def visualize_subset(
    graph,
    subset,
    color=None,
    width=5,
    is_proposal_subset=False,
    groundtruth_graph=None,
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
    width : int, optional
        Line width used to plot "subset". The default is 5.
    proposals_subset : bool, optional
        Indication of whether "subset" is a subset of proposals. The default
        is False.
    groundtruth_graph : networkx.Graph, optional
        Graph generated from ground truth tracings. The default is None.
    title : str, optional
        Title of the plot. Default is "Proposals".
    """
    # Plot graph
    data = get_edge_traces(graph, graph.edges, color="black")
    data.append(get_node_traces(graph))
    if is_proposal_subset:
        data.extend(
            get_proposal_traces(graph, subset, color=color, width=width)
        )
    else:
        data.extend(get_edge_traces(graph, subset, width=width))

    # Add target graph (if applicable)
    if groundtruth_graph:
        data.extend(
            visualize_connected_components(
                groundtruth_graph, return_data=True
            )
        )
    plot(data, title)


# --- Helpers ---
def get_node_traces(graph):
    trace = go.Scatter3d(
        x=graph.node_xyz[:, 0],
        y=graph.node_xyz[:, 1],
        z=graph.node_xyz[:, 2],
        mode="markers",
        name="Nodes",
        marker=dict(size=2, color="red"),
    )
    return trace


def get_proposal_traces(graph, proposals, color=None, width=5):
    # Set preferences
    if color is None:
        line = dict(width=width)
    else:
        line = dict(color=color, width=width)

    # Add traces
    traces = []
    for p in proposals:
        xyz = graph.proposal_attr(p, "xyz")
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


def get_edge_traces(graph, edges, color=None, width=3):
    traces = []
    line = (
        dict(width=5) if color is None else dict(color=color, width=width)
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
