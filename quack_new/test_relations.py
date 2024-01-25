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

    assert a in graph.get_in_nodes_with_relation_type_and_parameter(b, NonEquivalenceRelationType.KeyOf)
    assert b in graph.get_out_nodes_with_relation_type_and_parameter(a, NonEquivalenceRelationType.KeyOf)

    c = ast.AST()
    d = ast.AST()
    graph.add_relation(c, d, NonEquivalenceRelationType.ArgumentOf, 2)

    assert c in graph.get_in_nodes_with_relation_type_and_parameter(d, NonEquivalenceRelationType.ArgumentOf, 2)
    assert d in graph.get_out_nodes_with_relation_type_and_parameter(c, NonEquivalenceRelationType.ArgumentOf, 2)

    e = ast.AST()
    graph.add_relation(c, e, NonEquivalenceRelationType.ElementOf, 0)

    assert c in graph.get_in_nodes_with_relation_type_and_parameter(e, NonEquivalenceRelationType.ElementOf, 0)
    assert e in graph.get_out_nodes_with_relation_type_and_parameter(c, NonEquivalenceRelationType.ElementOf, 0)

    f = ast.AST()
    g = ast.AST()
    h = ast.AST()
    i = ast.AST()
    j = ast.AST()
    k = ast.AST()
    l = ast.AST()
    m = ast.AST()

    graph.add_relation(h, f, NonEquivalenceRelationType.KeyOf)
    graph.add_relation(i, f, NonEquivalenceRelationType.ValueOf)
    graph.add_relation(i, g, NonEquivalenceRelationType.ValueOf)
    graph.add_relation(j, g, NonEquivalenceRelationType.ValueOf)
    graph.add_relation(f, k, NonEquivalenceRelationType.ValueOf)
    graph.add_relation(f, l, NonEquivalenceRelationType.KeyOf)
    graph.add_relation(g, l, NonEquivalenceRelationType.KeyOf)
    graph.add_relation(g, m, NonEquivalenceRelationType.KeyOf)

    graph.merge_nodes(f, g)

    assert graph.nodes_to_relation_types_to_parameters_to_in_nodes[g] == {
        NonEquivalenceRelationType.KeyOf: {
            None: {
                h
            }
        },
        NonEquivalenceRelationType.ValueOf: {
            None: {
                i, j
            }
        }
    }

    assert graph.nodes_to_relation_types_to_parameters_to_out_nodes[g] == {
        NonEquivalenceRelationType.KeyOf: {
            None: {
                l, m
            }
        },
        NonEquivalenceRelationType.ValueOf: {
            None: {
                k
            }
        }
    }

    assert graph.nodes_to_relation_types_to_parameters_to_out_nodes[h] == {
        NonEquivalenceRelationType.KeyOf: {
            None: {
                g
            }
        },
    }

    assert graph.nodes_to_relation_types_to_parameters_to_out_nodes[i] == {
        NonEquivalenceRelationType.ValueOf: {
            None: {
                g
            }
        },
    }

    assert graph.nodes_to_relation_types_to_parameters_to_in_nodes[k] == {
        NonEquivalenceRelationType.ValueOf: {
            None: {
                g
            }
        },
    }

    assert graph.nodes_to_relation_types_to_parameters_to_in_nodes[l] == {
        NonEquivalenceRelationType.KeyOf: {
            None: {
                g
            }
        },
    }

    copy_graph = graph.copy()
    assert id(graph) != (copy_graph)

    assert copy_graph.nodes == graph.nodes
    assert id(copy_graph.nodes) != id(graph.nodes)

    assert copy_graph.nodes_to_relation_types_to_parameters_to_in_nodes == graph.nodes_to_relation_types_to_parameters_to_in_nodes
    assert id(copy_graph.nodes_to_relation_types_to_parameters_to_in_nodes) != id(graph.nodes_to_relation_types_to_parameters_to_in_nodes)

    assert copy_graph.nodes_to_relation_types_to_parameters_to_out_nodes == graph.nodes_to_relation_types_to_parameters_to_out_nodes
    assert id(copy_graph.nodes_to_relation_types_to_parameters_to_out_nodes) != id(graph.nodes_to_relation_types_to_parameters_to_out_nodes)
