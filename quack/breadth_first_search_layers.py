import typing

import networkx as nx


def breadth_first_search_layers(
        G: nx.DiGraph
) -> typing.Iterator[set[typing.Any]]:
    starting_node_set = {
        node
        for node, out_degree in G.out_degree
        if not out_degree
    }

    visited: set[typing.Any] = starting_node_set.copy()
    current_layer: set[typing.Any] = starting_node_set.copy()

    while current_layer:
        yield current_layer

        # Get the next layer of nodes
        next_layer: set[typing.Any] = set()
        for node in current_layer:
            for predecessor in G.predecessors(node):
                if predecessor not in visited:
                    visited.add(predecessor)
                    next_layer.add(predecessor)

        current_layer = next_layer
