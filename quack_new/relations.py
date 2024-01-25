"""
Relations among type variables.
"""
import _ast
import ast
import typing
from collections import defaultdict
from enum import Enum, auto

import networkx as nx


class EquivalenceRelationGraph(nx.Graph):
    def set_equivalent(self, from_: ast.AST, to: ast.AST):
        self.add_edge(from_, to)

    def get_equivalence_relations_among_nodes(self, node_set: typing.AbstractSet[ast.AST]) -> typing.Iterator[tuple[ast.AST]]:
        nodes_in_graph = node_set & self.nodes.keys()
        subgraph = nx.subgraph(self, nodes_in_graph)
        yield from subgraph.edges


class NonEquivalenceRelationType(Enum):
    KeyOf = auto()
    ValueOf = auto()
    IterTargetOf = auto()
    ArgumentOf = auto()
    ReturnedValueOf = auto()
    AttrOf = auto()
    ElementOf = auto()
    SendTargetOf = auto()
    YieldFromAwaitResultOf = auto()


NonEquivalenceRelationTuple = tuple[NonEquivalenceRelationType, ...]


class NonEquivalenceRelationGraph:
    def __init__(self):
        self.nodes: set[_ast.AST] = set()

        self.nodes_to_relation_types_to_parameters_to_out_nodes: defaultdict[
            _ast.AST,
            defaultdict[
                NonEquivalenceRelationType,
                defaultdict[
                    typing.Optional[object],
                    set[_ast.AST]
                ]
            ]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

        self.nodes_to_relation_types_to_parameters_to_in_nodes: defaultdict[
            _ast.AST,
            defaultdict[
                NonEquivalenceRelationType,
                defaultdict[
                    typing.Optional[object],
                    set[_ast.AST]
                ]
            ]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    def __contains__(self, item: _ast.AST) -> bool:
        return item in self.nodes

    def copy(self):
        new_graph = NonEquivalenceRelationGraph()

        new_graph.nodes.update(self.nodes)

        for node, relation_types_to_parameters_to_out_nodes in self.nodes_to_relation_types_to_parameters_to_out_nodes.items():
            for relation_type, parameters_to_out_nodes in relation_types_to_parameters_to_out_nodes.items():
                for parameter, out_nodes in parameters_to_out_nodes.items():
                    new_graph.nodes_to_relation_types_to_parameters_to_out_nodes[node][relation_type][parameter].update(out_nodes)

        for node, relation_types_to_parameters_to_in_nodes in self.nodes_to_relation_types_to_parameters_to_in_nodes.items():
            for relation_type, parameters_to_in_nodes in relation_types_to_parameters_to_in_nodes.items():
                for parameter, in_nodes in parameters_to_in_nodes.items():
                    new_graph.nodes_to_relation_types_to_parameters_to_in_nodes[node][relation_type][parameter].update(in_nodes)

        return new_graph

    def add_node(self, node: _ast.AST) -> None:
        self.nodes.add(node)

    def add_relation(
            self,
            from_: _ast.AST,
            to: _ast.AST,
            relation_type: NonEquivalenceRelationType,
            parameter: typing.Optional[object] = None
    ) -> None:
        self.nodes.add(from_)

        self.nodes.add(to)

        self.nodes_to_relation_types_to_parameters_to_out_nodes[from_][relation_type][parameter].add(to)
        self.nodes_to_relation_types_to_parameters_to_in_nodes[to][relation_type][parameter].add(from_)

    def has_relation(
            self,
            from_node: _ast.AST,
            to_node: _ast.AST,
            relation_type: NonEquivalenceRelationType,
            parameter: typing.Optional[object] = None
    ) -> bool:
        return to_node in self.nodes_to_relation_types_to_parameters_to_out_nodes[from_node][relation_type][parameter]

    def get_in_nodes_with_relation_type_and_parameter(self, to: _ast.AST, relation_type: NonEquivalenceRelationType, parameter: typing.Optional[object] = None) -> set[_ast.AST]:
        if to in self.nodes:
            relation_types_to_parameters_to_in_nodes = self.nodes_to_relation_types_to_parameters_to_in_nodes[to]
            if relation_type in relation_types_to_parameters_to_in_nodes:
                parameters_to_in_nodes = relation_types_to_parameters_to_in_nodes[relation_type]
                if parameter in parameters_to_in_nodes:
                    return parameters_to_in_nodes[parameter]

        return set()

    def get_out_nodes(self, from_: _ast.AST) -> dict[NonEquivalenceRelationType, dict[object, set[_ast.AST]]]:
        if from_ in self.nodes:
            # Return a copy to prevent modification of the graph
            return {
                relation_type: {
                    parameter: {
                        out_node
                        for out_node in out_nodes
                    }
                    for parameter, out_nodes in parameters_to_out_nodes.items()
                }
                for relation_type, parameters_to_out_nodes in self.nodes_to_relation_types_to_parameters_to_out_nodes[from_].items()
            }

        return dict()

    def get_out_nodes_with_relation_type(self, from_: _ast.AST, relation_type: NonEquivalenceRelationType) -> dict[object, set[_ast.AST]]:
        if from_ in self.nodes:
            relation_types_to_parameters_to_out_nodes = self.nodes_to_relation_types_to_parameters_to_out_nodes[from_]
            if relation_type in relation_types_to_parameters_to_out_nodes:
                # Return a copy to prevent modification of the graph
                return {
                    parameter: {
                        out_node
                        for out_node in out_nodes
                    }
                    for parameter, out_nodes in relation_types_to_parameters_to_out_nodes[relation_type].items()
                }

        return dict()

    def get_out_nodes_with_relation_type_and_parameter(self, from_: _ast.AST, relation_type: NonEquivalenceRelationType, parameter: typing.Optional[object] = None) -> set[_ast.AST]:
        if from_ in self.nodes:
            relation_types_to_parameters_to_out_nodes = self.nodes_to_relation_types_to_parameters_to_out_nodes[from_]
            if relation_type in relation_types_to_parameters_to_out_nodes:
                parameters_to_out_nodes = relation_types_to_parameters_to_out_nodes[relation_type]
                if parameter in parameters_to_out_nodes:
                    # Return a copy to prevent modification of the graph
                    return parameters_to_out_nodes[parameter].copy()

        return set()

    def get_all_relation_types_and_parameters(self, from_: _ast.AST) -> list[tuple[NonEquivalenceRelationType, typing.Optional[object]]]:
        relation_types_and_parameters = list()

        if from_ in self.nodes:
            relation_types_to_parameters_to_out_nodes = self.nodes_to_relation_types_to_parameters_to_out_nodes[from_]
            for relation_type, parameters_to_out_nodes in relation_types_to_parameters_to_out_nodes.items():
                for parameter in parameters_to_out_nodes.keys():
                    relation_types_and_parameters.append((relation_type, parameter))

        return relation_types_and_parameters

    def merge_nodes(self, target_node: _ast.AST, acquirer_node: _ast.AST):
        if target_node != acquirer_node:
            self.add_node(target_node)
            self.add_node(acquirer_node)

            # For each in-edge (in_node, relation_type, parameter) of target_node,
            # Identify (in_node, relation_type, parameter) triples such that (in_node, relation_type, parameter) is not an in-edge of acquirer_node
            # Add (in_node, relation_type, parameter) as an in-edge of acquirer_node,
            # (relation_type, parameter, acquirer_node) as an out-edge of in_node.
            in_node_relation_type_parameter_triples_to_deal_with: set[tuple[_ast.AST, NonEquivalenceRelationType, typing.Optional[object]]] = set()

            for relation_type, parameters_to_in_nodes in self.nodes_to_relation_types_to_parameters_to_in_nodes[target_node].items():
                for parameter, in_nodes in parameters_to_in_nodes.items():
                    for in_node in in_nodes:
                        if in_node not in self.nodes_to_relation_types_to_parameters_to_in_nodes[acquirer_node][relation_type][parameter]:
                            in_node_relation_type_parameter_triples_to_deal_with.add((in_node, relation_type, parameter))

            for in_node, relation_type, parameter in in_node_relation_type_parameter_triples_to_deal_with:
                self.nodes_to_relation_types_to_parameters_to_in_nodes[acquirer_node][relation_type][parameter].add(in_node)
                self.nodes_to_relation_types_to_parameters_to_out_nodes[in_node][relation_type][parameter].add(acquirer_node)

            # For each out-edge (relation_type, parameter, out_node) of target_node,
            # Identify (relation_type, parameter, out_node) triples such that (relation_type, parameter, out_node) is not an out-edge of acquirer_node
            # Add (relation_type, parameter, out_node) as an out-edge of acquirer_node,
            # (acquirer_node, relation_type, parameter) as an in-edge of out_node,
            relation_type_parameter_out_node_triples_to_deal_with: set[tuple[NonEquivalenceRelationType, typing.Optional[object], _ast.AST]] = set()

            for relation_type, parameters_to_out_nodes in self.nodes_to_relation_types_to_parameters_to_out_nodes[target_node].items():
                for parameter, out_nodes in parameters_to_out_nodes.items():
                    for out_node in out_nodes:
                        if out_node not in self.nodes_to_relation_types_to_parameters_to_out_nodes[acquirer_node][relation_type][parameter]:
                            relation_type_parameter_out_node_triples_to_deal_with.add((relation_type, parameter, out_node))

            for relation_type, parameter, out_node in relation_type_parameter_out_node_triples_to_deal_with:
                self.nodes_to_relation_types_to_parameters_to_out_nodes[acquirer_node][relation_type][parameter].add(out_node)
                self.nodes_to_relation_types_to_parameters_to_in_nodes[out_node][relation_type][parameter].add(acquirer_node)

            # Remove target_node from the graph
            self.nodes.remove(target_node)

            for relation_type, parameters_to_in_nodes in self.nodes_to_relation_types_to_parameters_to_in_nodes[target_node].items():
                for parameter, in_nodes in parameters_to_in_nodes.items():
                    for in_node in in_nodes:
                        self.nodes_to_relation_types_to_parameters_to_out_nodes[in_node][relation_type][parameter].remove(target_node)

            del self.nodes_to_relation_types_to_parameters_to_in_nodes[target_node]

            for relation_type, parameters_to_out_nodes in self.nodes_to_relation_types_to_parameters_to_out_nodes[target_node].items():
                for parameter, out_nodes in parameters_to_out_nodes.items():
                    for out_node in out_nodes:
                        self.nodes_to_relation_types_to_parameters_to_in_nodes[out_node][relation_type][parameter].remove(target_node)

            del self.nodes_to_relation_types_to_parameters_to_out_nodes[target_node]

            return in_node_relation_type_parameter_triples_to_deal_with | {
                (acquirer_node, relation_type, parameter)
                for relation_type, parameter, out_node in relation_type_parameter_out_node_triples_to_deal_with
            }
        else:
            import pudb
            pudb.set_trace()
            return set()
