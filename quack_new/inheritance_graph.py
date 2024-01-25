import collections
import typing

import networkx

from type_definitions import RuntimeClass


def construct_base_to_derived_graph(runtime_class_set: typing.AbstractSet[RuntimeClass]) -> networkx.DiGraph:
    base_to_derived_graph: networkx.DiGraph = networkx.DiGraph()

    for runtime_class in runtime_class_set:
        base_to_derived_graph.add_node(runtime_class)

        for base_class in runtime_class.__bases__:
            if base_class in runtime_class_set:
                base_to_derived_graph.add_edge(base_class, runtime_class)

    return base_to_derived_graph
