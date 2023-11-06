import typing

import networkx as nx


def graph_condensation(
        digraph: nx.DiGraph,
):
    """
    Condense a directed graph into a directed acyclic graph (DAG) by
    contracting strongly connected components (SCCs) into a single node.

    :param digraph: A directed graph.
    :return: A directed acyclic graph (DAG).
    """
    strongly_connected_component_list: list[set[typing.Any]] = list(nx.strongly_connected_components(digraph))

    node_to_strongly_connected_component_index_dict: dict[typing.Any, int] = {
        node: strongly_connected_component_index
        for strongly_connected_component_index, strongly_connected_component in enumerate(strongly_connected_component_list)
        for node in strongly_connected_component
    }
    condensed_graph = nx.DiGraph()
    for node in digraph.nodes:
        condensed_graph.add_node(node_to_strongly_connected_component_index_dict[node])
    for (start, end) in digraph.edges:
        start_strongly_connected_component_index = node_to_strongly_connected_component_index_dict[start]
        end_strongly_connected_component_index = node_to_strongly_connected_component_index_dict[end]

        if start_strongly_connected_component_index != end_strongly_connected_component_index:
            condensed_graph.add_edge(
                start_strongly_connected_component_index,
                end_strongly_connected_component_index,
            )

    return condensed_graph, strongly_connected_component_list, node_to_strongly_connected_component_index_dict
