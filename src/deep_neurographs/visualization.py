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
    graph, line_width=4, return_data=False, title=""
):
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


def visualize_graph(graph, title="Initial Segmentation"):
    data = plot_edges(graph, graph.edges)
    data.append(plot_nodes(graph))
    plot(data, title)


def visualize_proposals(graph, title="Edge Proposals"):
    visualize_subset(graph, graph.proposals, proposal_subset=True, title=title)


def visualize_targets(graph, target_graph=None, title="Target Edges"):
    visualize_subset(
        graph,
        graph.target_edges,
        proposal_subset=True,
        target_graph=target_graph,
        title=title,
    )


def visualize_subset(
    graph,
    edges,
    line_width=5,
    proposal_subset=False,
    target_graph=None,
    title="",
):
    # Plot graph
    data = plot_edges(graph, graph.edges, color="black")
    data.append(plot_nodes(graph))
    if proposal_subset:
        data.extend(plot_proposals(graph, edges, line_width=line_width))
    else:
        data.extend(plot_edges(graph, edges, line_width=line_width))

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
        marker=dict(size=3, color="red"),
    )


def plot_proposals(graph, proposals, color=None, line_width=3.5):
    traces = []
    line = (
        dict(width=5) if color is None else dict(color=color, width=line_width)
    )
    for edge in proposals:
        trace = go.Scatter3d(
            x=graph.proposals[edge]["xyz"][:, 0],
            y=graph.proposals[edge]["xyz"][:, 1],
            z=graph.proposals[edge]["xyz"][:, 2],
            mode="lines",
            line=line,
            name="{}".format(edge),
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
            x=graph.edges[(i, j)]["xyz"][:, 0],
            y=graph.edges[(i, j)]["xyz"][:, 1],
            z=graph.edges[(i, j)]["xyz"][:, 2],
            mode="lines",
            line=line,
            name="({},{})".format(i, j),
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
