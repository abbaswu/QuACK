import typing

import networkx as nx


def breadth_first_search_layers(
        G: nx.DiGraph,
        starting_node_frozenset: typing.Optional[frozenset[typing.Any]] = None
) -> typing.Iterator[set[typing.Any]]:
    if starting_node_frozenset is None:
        starting_node_set = {
            node
            for node, in_degree in G.in_degree
            if not in_degree
        }
    else:
        starting_node_set = set(starting_node_frozenset)

    visited: set[typing.Any] = starting_node_set.copy()
    current_layer: set[typing.Any] = starting_node_set.copy()

    while current_layer:
        yield current_layer

        # Get the next layer of nodes
        next_layer: set[typing.Any] = set()
        for node in current_layer:
            if node in G.nodes:
                for succ in G.successors(node):
                    if succ not in visited:
                        visited.add(succ)
                        next_layer.add(succ)

        current_layer = next_layer
