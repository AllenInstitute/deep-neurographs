"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


"""

import networkx as nx
import numpy as np
import plotly.graph_objects as go


def visualize_connected_components(graph):
    pass


def visualize_immutables(graph, title="Immutable Graph"):
    data = plot_edges(graph, graph.immutable_edges)
    data.append(plot_nodes(graph))
    plot(data, title)


def visualize_proposals(graph, title="Edge Proposals"):
    visualize_subset(graph, graph.mutable_edges, title=title)


def visualize_targets(graph, target_graph=None, title="Target Edges"):
    visualize_subset(
        graph,
        graph.target_edges,
        target_graph=target_graph,
        title=title,
    )


def visualize_subset(graph, edges, target_graph=None, title=""):
    data = plot_edges(graph, graph.immutable_edges, color="black")
    data.extend(plot_edges(graph, edges))
    data.append(plot_nodes(graph)) 
    if target_graph:
        edges = target_graph.immutable_edges
        data.extend(plot_edges(target_graph, edges, color="blue"))
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


def plot_edges(graph, edges, color=None):
    traces = []
    line = dict(width=4) if color is None else dict(color=color, width=3)
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
        plot_bgcolor="white",
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )
    fig.update_layout(
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=1)),
        width=1200,
        height=600,
    )
    fig.show()


def subplot(data1, data2, title):
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]]
    )
    fig.add_trace(data1, row=1, col=1)
    fig.add_trace(data2, row=1, col=2)
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

    fig.update_layout(width=1200, height=800)
    fig.show()
