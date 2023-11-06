import collections
import typing

import networkx

from type_definitions import RuntimeClass


InheritanceGraph = networkx.DiGraph


def construct_inheritance_graph(runtime_class_set: typing.AbstractSet[RuntimeClass]) -> InheritanceGraph:
    inheritance_graph: InheritanceGraph = InheritanceGraph()

    for runtime_class in runtime_class_set:
        inheritance_graph.add_node(runtime_class)

        for base_class in runtime_class.__bases__:
            if base_class in runtime_class_set:
                inheritance_graph.add_edge(runtime_class, base_class)

    return inheritance_graph


def iterate_inheritance_graph(
        inheritance_graph: InheritanceGraph
) -> typing.Iterator[RuntimeClass]:

    # Do a breadth-first search
    starting_node_set = {
        node
        for node, out_degree in inheritance_graph.out_degree
        if not out_degree
    }

    visited: set[RuntimeClass] = starting_node_set.copy()

    def recursive_function(node: RuntimeClass):
        nonlocal visited

        yield node

        for predecessor in inheritance_graph.predecessors(node):
            if predecessor not in visited:
                visited.add(predecessor)
                yield from recursive_function(predecessor)

    for starting_node in starting_node_set:
        yield from recursive_function(starting_node)
