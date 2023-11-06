"""
Generated with ChatGPT:
https://chat.openai.com/share/c12db936-d4fa-494c-a852-09dd08caca3d

To generate an HTML coverage report:

- Run Coverage: coverage run --source=relations test_relations.py
- Generate an HTML report: coverage html

For writing tests for the OtherRelationsGraph class, we need to cover the following:

    Basic graph operations: initialization, adding relations, and copying the graph.
    Retrieval operations: get or create related nodes and get in/out edges based on relation tuples/types.
    Edge cases: handling nodes that don't exist, using parameters or not, etc.

Here's a basic test suite:
"""

import ast
import unittest
from collections import defaultdict
from relations import NonEquivalenceRelationType, NonEquivalenceRelationGraph

if __name__ == '__main__':
    graph = NonEquivalenceRelationGraph()

    a, b = ast.AST(), ast.AST()
    graph.add_relation(a, b, NonEquivalenceRelationType.KeyOf)

    assert a in graph.get_in_edges_by_relation_tuple(b)[(NonEquivalenceRelationType.KeyOf,)]
    assert b in graph.get_out_edges_by_relation_tuple(a)[(NonEquivalenceRelationType.KeyOf,)]
    assert b == graph.get_or_create_related_node(a, NonEquivalenceRelationType.KeyOf)

    # Retrieving the related node
    related_node = graph.get_or_create_related_node(a, NonEquivalenceRelationType.ValueOf)
    assert related_node != a != b
    assert a in graph.get_in_edges_by_relation_tuple(related_node)[(NonEquivalenceRelationType.ValueOf,)]
    assert related_node in graph.get_out_edges_by_relation_tuple(a)[(NonEquivalenceRelationType.ValueOf,)]
    assert related_node == graph.get_or_create_related_node(a, NonEquivalenceRelationType.ValueOf)

    related_node = graph.get_or_create_related_node(a, NonEquivalenceRelationType.ValueOf)

    assert related_node != a != b

    assert a in graph.get_in_edges_by_relation_tuple(related_node)[(NonEquivalenceRelationType.ValueOf,)]
    assert related_node in graph.get_out_edges_by_relation_tuple(a)[(NonEquivalenceRelationType.ValueOf,)]

    c = ast.AST()
    d = graph.get_or_create_related_node(c, NonEquivalenceRelationType.ArgumentOf, 2)

    assert c in graph.get_in_edges_by_relation_tuple(d)[(NonEquivalenceRelationType.ArgumentOf, 2)]
    assert d in graph.get_out_edges_by_relation_tuple(c)[(NonEquivalenceRelationType.ArgumentOf, 2)]
    assert d == graph.get_or_create_related_node(c, NonEquivalenceRelationType.ArgumentOf, 2)

    e = ast.AST()

    graph.add_relation(c, e, NonEquivalenceRelationType.ElementOf, 0)

    assert c in graph.get_in_edges_by_relation_tuple(e)[(NonEquivalenceRelationType.ElementOf, 0)]
    assert e in graph.get_out_edges_by_relation_tuple(c)[(NonEquivalenceRelationType.ElementOf, 0)]
    assert e == graph.get_or_create_related_node(c, NonEquivalenceRelationType.ElementOf, 0)

    copy_graph = graph.copy()

    assert all(
        node in copy_graph.digraph.nodes
        for node in graph.digraph.nodes
    )

    assert all(
        edge in copy_graph.digraph.edges
        for edge in graph.digraph.edges
    )

    assert id(graph) != (copy_graph)
    assert id(copy_graph.digraph) != id(graph.digraph)
