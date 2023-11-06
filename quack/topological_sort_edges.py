import typing

import networkx as nx


def topological_sort_edges(digraph: nx.DiGraph):
    # Perform the topological sort on nodes
    # Create a mapping from node to its position in the topological order
    node_position: dict[typing.Any, int] = {
        node: pos for pos, node in enumerate(nx.topological_sort(digraph))
    }

    # Sort the edges based on the topological order of their source nodes
    topological_order_edges: list[tuple[typing.Any, typing.Any]] = sorted(
        digraph.edges(data=False), key=lambda edge: node_position[edge[0]]
    )

    return topological_order_edges


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edges_from([
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (4, 6),
        (5, 6),
    ])
