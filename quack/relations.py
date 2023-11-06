"""
Relations among type variables.
"""
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
    def __init__(self, digraph: typing.Optional[nx.DiGraph] = None):
        if digraph is not None:
            self.digraph = digraph
        else:
            self.digraph = nx.DiGraph()

    def add_node(self, node: ast.AST):
        self.digraph.add_node(node)

    def add_relation(self, from_: ast.AST, to: ast.AST, relation_type: NonEquivalenceRelationType,
                     parameter: typing.Optional[object] = None) -> None:
        self.digraph.add_edge(from_, to)

        if parameter is None:
            relation_tuple = (relation_type,)
        else:
            relation_tuple = (relation_type, parameter)

        self.digraph.edges[from_, to].setdefault(relation_tuple)

    def get_or_create_related_node(self, from_: ast.AST, relation_type: NonEquivalenceRelationType,
                                   parameter: typing.Optional[object] = None) -> ast.AST:
        if parameter is None:
            relation_tuple = (relation_type,)
        else:
            relation_tuple = (relation_type, parameter)

        if from_ in self.digraph:
            atlas_view = self.digraph[from_]

            related_node: ast.AST | None = next(
                (k for k, v in atlas_view.items() if relation_tuple in v),
                None
            )

            if related_node is None:
                # No related node available. Create one and return it.
                dummy_node = ast.AST()

                self.digraph.add_edge(from_, dummy_node)
                self.digraph.edges[from_, dummy_node].setdefault(relation_tuple)

                return dummy_node
            else:
                # Related node found. Return it.
                return related_node
        else:
            dummy_node = ast.AST()

            self.digraph.add_edge(from_, dummy_node)
            self.digraph.edges[from_, dummy_node].setdefault(relation_tuple)

            return dummy_node

    def get_or_create_inversely_related_node(self, to: ast.AST, relation_type: NonEquivalenceRelationType,
                                             parameter: typing.Optional[object] = None) -> ast.AST:
        if parameter is None:
            relation_tuple = (relation_type,)
        else:
            relation_tuple = (relation_type, parameter)

        if to in self.digraph:
            inversely_related_node: ast.AST | None = next(
                (from_ for from_, _, data in self.digraph.in_edges(nbunch=to, data=True) if relation_tuple in data),
                None
            )

            if inversely_related_node is None:
                # No inversely related node available. Create one and return it.
                dummy_node = ast.AST()

                self.digraph.add_edge(dummy_node, to)
                self.digraph.edges[dummy_node, to].setdefault(relation_tuple)

                return dummy_node
            else:
                # Inversely related node found. Return it.
                return inversely_related_node
        else:
            dummy_node = ast.AST()

            self.digraph.add_edge(dummy_node, to)
            self.digraph.edges[dummy_node, to].setdefault(relation_tuple)

            return dummy_node

    def iterate_relations(self) -> typing.Iterator[tuple[ast.AST, ast.AST, NonEquivalenceRelationTuple]]:
        for from_, to, relation_tuples in self.digraph.edges(data=True):
            for relation_tuple in relation_tuples:
                yield from_, to, relation_tuple

    def copy(self):
        return NonEquivalenceRelationGraph(self.digraph.copy())

    def get_in_edges_by_relation_tuple(self, to: ast.AST) -> defaultdict[NonEquivalenceRelationTuple, set[ast.AST]]:
        in_edges_by_relation_tuple: defaultdict[NonEquivalenceRelationTuple, set[ast.AST]] = defaultdict(set)

        if to in self.digraph.nodes:
            for from_, _, relation_tuples in self.digraph.in_edges(to, data=True):
                for relation_tuple in relation_tuples:
                    in_edges_by_relation_tuple[relation_tuple].add(from_)

        return in_edges_by_relation_tuple

    def get_out_edges_by_relation_tuple(self, from_: ast.AST) -> defaultdict[NonEquivalenceRelationTuple, set[ast.AST]]:
        out_edges_by_relation_tuple: defaultdict[NonEquivalenceRelationTuple, set[ast.AST]] = defaultdict(set)

        if from_ in self.digraph.nodes:
            for _, to, relation_tuples in self.digraph.out_edges(from_, data=True):
                for relation_tuple in relation_tuples:
                    out_edges_by_relation_tuple[relation_tuple].add(to)

        return out_edges_by_relation_tuple

    def get_all_out_edges_by_relation_tuple(self, froms: typing.AbstractSet[ast.AST]):
        all_out_edges_by_relation_tuple: defaultdict[NonEquivalenceRelationTuple, set[ast.AST]] = defaultdict(set)

        for from_ in froms:
            if from_ in self.digraph.nodes:
                for _, to, relation_tuples in self.digraph.out_edges(from_, data=True):
                    for relation_tuple in relation_tuples:
                        all_out_edges_by_relation_tuple[relation_tuple].add(to)

        return all_out_edges_by_relation_tuple

    def iterate_nodes_and_out_edges_by_relation_tuple(self) -> typing.Iterator[
        tuple[ast.AST, defaultdict[NonEquivalenceRelationTuple, set[ast.AST]]]]:
        for node in self.digraph.nodes:
            out_edges_by_relation_tuple: defaultdict[NonEquivalenceRelationTuple, set[ast.AST]] = defaultdict(set)

            for _, to, relation_tuples in self.digraph.out_edges(node, data=True):
                for relation_tuple in relation_tuples:
                    out_edges_by_relation_tuple[relation_tuple].add(to)

            yield node, out_edges_by_relation_tuple

    def remove_node(self, node: ast.AST) -> None:
        if node in self.digraph.nodes:
            self.digraph.remove_node(node)
