import _ast
import ast
import asyncio
import itertools
import json
import logging
import typing

from collections import defaultdict, Counter
from types import ModuleType, FunctionType

import numpy as np

from determine_number_of_type_variables import determine_number_of_type_variables
from disjoint_set import DisjointSet
from get_attributes_in_runtime_class import get_attributes_in_runtime_class
from relations import NonEquivalenceRelationGraph, NonEquivalenceRelationTuple, NonEquivalenceRelationType
from runtime_term import RuntimeTerm, Instance
from type_ascription import type_ascription
from type_definitions import RuntimeClass
from class_query_database import ClassQueryDatabase
from typeshed_client_ex.client import Client

from typeshed_client_ex.type_definitions import TypeshedTypeAnnotation, TypeshedClass, from_runtime_class, \
    TypeshedTypeVariable, subscribe, replace_type_variables_in_type_annotation, Subscription


def dump_confidence_and_possible_class_list(
    confidence_and_possible_class_list: list[tuple[float, TypeshedClass]],
    class_inference_log_file_io: typing.IO
):
    confidence_and_possible_class_string_list_list: list[list[float, str]] = [
        [confidence, str(possible_class)]
        for confidence, possible_class in confidence_and_possible_class_list
    ]

    json.dump(confidence_and_possible_class_string_list_list, class_inference_log_file_io)
    class_inference_log_file_io.write('\n')


class TypeInference:
    def __init__(
            self,
            node_disjoint_set: DisjointSet[_ast.AST],
            equivalent_node_disjoint_set_top_nodes_to_attribute_counters: defaultdict[ast.AST, Counter[str]],
            equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph: NonEquivalenceRelationGraph,
            runtime_term_sharing_node_disjoint_set: DisjointSet[_ast.AST],
            runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]],
            type_query_database: ClassQueryDatabase,
            client: Client
    ):
        self.node_disjoint_set = node_disjoint_set
        self.equivalent_node_disjoint_set_top_nodes_to_attribute_counters = equivalent_node_disjoint_set_top_nodes_to_attribute_counters
        self.equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph = equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph
        self.runtime_term_sharing_node_disjoint_set = runtime_term_sharing_node_disjoint_set
        self.runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets = runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets
        self.type_query_database = type_query_database
        self.client = client

        self.class_inference_cache: dict[
            frozenset[ast.AST],
            tuple[
                list[tuple[float, TypeshedTypeAnnotation]],
                bool
            ]
        ] = dict()
        self.type_inference_cache: dict[frozenset[ast.AST], TypeshedTypeAnnotation] = dict()

    def infer_classes_for_equivalent_node_disjoint_set_top_nodes(
            self,
            equivalent_node_disjoint_set_top_nodes: frozenset[ast.AST],
            indent_level: int = 0,
            cosine_similarity_threshold: float = 1e-1
    ) -> tuple[
        list[tuple[float, TypeshedClass]],  # class inference confidences and classs
        bool  # whether runtime class can be instance-of types.NoneType
    ]:
        indent = '    ' * indent_level

        # Has a record in cache
        if equivalent_node_disjoint_set_top_nodes in self.class_inference_cache:
            logging.info(
                '%sCache hit when performing class inference for %s.',
                indent,
                equivalent_node_disjoint_set_top_nodes
            )

            return self.class_inference_cache[equivalent_node_disjoint_set_top_nodes]
        else:
            # No record in cache
            logging.info(
                '%sCache miss when performing class inference for %s.',
                indent,
                equivalent_node_disjoint_set_top_nodes
            )

            # Determine whether it can be None.

            can_be_none: bool = False

            runtime_term_sharing_node_disjoint_set_top_node_set = {
                self.runtime_term_sharing_node_disjoint_set.find(equivalent_node)
                for node in equivalent_node_disjoint_set_top_nodes
                for equivalent_node in self.node_disjoint_set.get_containing_set(node)
            }

            for runtime_term_sharing_node_disjoint_set_top_node in runtime_term_sharing_node_disjoint_set_top_node_set:
                if Instance(type(None)) in self.runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets[
                    runtime_term_sharing_node_disjoint_set_top_node
                ]:
                    can_be_none = True

            logging.info(
                '%sCan %s be None? %s',
                indent,
                equivalent_node_disjoint_set_top_nodes, can_be_none
            )

            # Initialize attribute counter.

            attribute_counter: Counter[str] = Counter()
            for node in equivalent_node_disjoint_set_top_nodes:
                attribute_counter.update(self.equivalent_node_disjoint_set_top_nodes_to_attribute_counters[node])

            logging.info(
                '%sAttribute counter for %s: %s',
                indent,
                equivalent_node_disjoint_set_top_nodes,
                attribute_counter
            )

            # Query possible classes.

            confidence_and_possible_class_list: list[tuple[float, TypeshedClass]] = list()

            (
                possible_class_ndarray,
                cosine_similarity_ndarray,
            ) = self.type_query_database.query(attribute_counter)

            nonzero_cosine_similarity_indices = (cosine_similarity_ndarray > cosine_similarity_threshold)

            selected_possible_class_ndarray = possible_class_ndarray[nonzero_cosine_similarity_indices]
            selected_cosine_similarity_ndarray = cosine_similarity_ndarray[nonzero_cosine_similarity_indices]

            argsort = np.argsort(selected_cosine_similarity_ndarray)

            for i in argsort[-1::-1]:
                possible_class = selected_possible_class_ndarray[i]
                cosine_similarity = float(selected_cosine_similarity_ndarray[i])

                confidence_and_possible_class_list.append(
                    (cosine_similarity, possible_class)
                )

            logging.info(
                '%sPossible types queried for %s based on attributes: %s',
                indent,
                equivalent_node_disjoint_set_top_nodes,
                confidence_and_possible_class_list
            )

            return_value = confidence_and_possible_class_list, can_be_none

            self.class_inference_cache[equivalent_node_disjoint_set_top_nodes] = return_value

            return return_value

    def infer_type_for_equivalent_node_disjoint_set_top_nodes(
            self,
            equivalent_node_disjoint_set_top_nodes: frozenset[ast.AST],
            depth: int = 0,
            cosine_similarity_threshold: float = 1e-1,
            depth_limit: int = 3,
            first_level_class_inference_failed_fallback: TypeshedClass = TypeshedClass('typing', 'Any'),
            non_first_level_class_inference_failed_fallback: TypeshedClass = TypeshedClass('typing', 'Any'),
            pre_class_inference_callback: typing.Optional[typing.Callable[[frozenset[ast.AST], int], None]] = None,
            class_inference_log_file_io: typing.Optional[typing.IO] = None
    ) -> TypeshedTypeAnnotation:
        indent = '    ' * depth

        # Has a record in cache
        if equivalent_node_disjoint_set_top_nodes in self.type_inference_cache:
            logging.info(
                '%sCache hit when performing type inference for %s.',
                indent,
                equivalent_node_disjoint_set_top_nodes
            )

            return self.type_inference_cache[equivalent_node_disjoint_set_top_nodes]
        else:
            # No record in cache
            logging.info(
                '%sCache miss when performing type inference for %s.',
                indent,
                equivalent_node_disjoint_set_top_nodes
            )

            if depth > depth_limit:
                logging.error(
                    '%sRecursive type inference exceeded depth limit of %s. Returning %s.',
                    indent,
                    depth_limit,
                    non_first_level_class_inference_failed_fallback
                )

                return_value = non_first_level_class_inference_failed_fallback
            else:
                # Part 1: Infer possible classes.
                logging.info(
                    '%sPerforming class inference for %s.',
                    indent,
                    equivalent_node_disjoint_set_top_nodes
                )

                if pre_class_inference_callback is not None:
                    pre_class_inference_callback(equivalent_node_disjoint_set_top_nodes, depth + 1)

                (
                    confidence_and_possible_class_list,
                    can_be_none
                ) = self.infer_classes_for_equivalent_node_disjoint_set_top_nodes(
                    equivalent_node_disjoint_set_top_nodes,
                    depth + 1,
                    cosine_similarity_threshold
                )

                if class_inference_log_file_io is not None:
                    dump_confidence_and_possible_class_list(
                        confidence_and_possible_class_list,
                        class_inference_log_file_io
                    )

                # Part 2: Infer type variables for possible classes to get final type inference results.
                if not confidence_and_possible_class_list:
                    top_class_prediction = (
                        first_level_class_inference_failed_fallback
                        if depth == 0
                        else non_first_level_class_inference_failed_fallback
                    )

                    logging.info(
                        '%sNo possible classes queried for %s based on attributes. Using %s.',
                        indent,
                        equivalent_node_disjoint_set_top_nodes,
                        top_class_prediction
                    )
                else:
                    (
                        top_class_prediction_confidence,
                        top_class_prediction
                    ) = confidence_and_possible_class_list[0]

                    logging.info(
                        '%sTop class prediction: %s',
                        indent,
                        top_class_prediction
                    )

                # Determine number of type variables.
                number_of_type_variables: int = determine_number_of_type_variables(
                    equivalent_node_disjoint_set_top_nodes,
                    top_class_prediction,
                    self.equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph,
                    self.client
                )

                logging.info(
                    '%sNumber of type variables for %s: %s',
                    indent,
                    top_class_prediction,
                    number_of_type_variables
                )

                if number_of_type_variables:
                    # Figure out what nodes are associated with each type variable.
                    type_variable_list: list[TypeshedTypeVariable] = [
                        TypeshedTypeVariable()
                        for _ in range(number_of_type_variables)
                    ]

                    type_variables_to_associated_node_sets: dict[TypeshedTypeVariable, set[ast.AST]] = {
                        type_variable: set()
                        for type_variable in type_variable_list
                    }

                    top_class_prediction_subscribed_by_type_variables: TypeshedTypeAnnotation = subscribe(
                        top_class_prediction,
                        tuple(type_variable_list)
                    )

                    async def get_related_nodes(
                            nodes: frozenset[ast.AST],
                            relation_type: NonEquivalenceRelationType,
                            parameter: typing.Optional[object] = None
                    ) -> frozenset[ast.AST]:
                        out_edges: set[ast.AST] = set()

                        for node in nodes:
                            out_edges.update(
                                self.equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph.get_out_nodes_with_relation_type_and_parameter(
                                    node,
                                    relation_type,
                                    parameter
                                )
                            )

                        return frozenset(out_edges)

                    async def typeshed_type_variable_ascription_callback(
                            nodes: frozenset[ast.AST],
                            typeshed_type_variable: TypeshedTypeVariable,
                            weight: float
                    ):
                        if typeshed_type_variable in type_variables_to_associated_node_sets:
                            type_variables_to_associated_node_sets[typeshed_type_variable].update(nodes)

                    async def typeshed_class_ascription_callback(
                            nodes: frozenset[ast.AST],
                            typeshed_class: TypeshedClass,
                            weight: float
                    ):
                        pass

                    asyncio.run(type_ascription(
                        equivalent_node_disjoint_set_top_nodes,
                        top_class_prediction_subscribed_by_type_variables,
                        get_related_nodes,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback
                    ))

                    # Infer type variables.

                    type_variables_to_type_inference_results: dict[
                        TypeshedTypeVariable,
                        TypeshedTypeAnnotation
                    ] = {
                        type_variable: (
                            first_level_class_inference_failed_fallback
                            if depth == 0
                            else non_first_level_class_inference_failed_fallback
                        )
                        for type_variable in type_variable_list
                    }

                    # Special handling for typing.Callable.
                    # If Ellipsis is a runtime term of the first argument, then that argument represents all arguments.
                    if (
                            top_class_prediction == TypeshedClass('typing', 'Callable')
                        and number_of_type_variables >= 2
                        and any((
                            Instance(type(Ellipsis)) in self.runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets[first_argument_node]
                            for first_argument_node
                            in type_variables_to_associated_node_sets[
                                type_variable_list[0]
                            ]
                        ))
                    ):
                        first_argument_type_inference_result: TypeshedClass = from_runtime_class(type(Ellipsis))

                        return_value_type_inference_result = self.infer_type_for_equivalent_node_disjoint_set_top_nodes(
                            frozenset(type_variables_to_associated_node_sets[
                                type_variable_list[-1]
                            ]),
                            depth + 1,
                            cosine_similarity_threshold,
                            depth_limit,
                            first_level_class_inference_failed_fallback,
                            non_first_level_class_inference_failed_fallback,
                            pre_class_inference_callback
                        )

                        type_inference_result = subscribe(
                            TypeshedClass('typing', 'Callable'),
                            (
                                first_argument_type_inference_result,
                                return_value_type_inference_result
                            )
                        )
                    else:
                        for type_variable, associated_node_set in type_variables_to_associated_node_sets.items():
                            logging.info(
                                '%sType variable: %s Associated node set: %s',
                                indent,
                                type_variable,
                                associated_node_set
                            )

                            assert not equivalent_node_disjoint_set_top_nodes & associated_node_set

                            top_inference_result = self.infer_type_for_equivalent_node_disjoint_set_top_nodes(
                                frozenset(associated_node_set),
                                depth + 1,
                                cosine_similarity_threshold,
                                depth_limit,
                                first_level_class_inference_failed_fallback,
                                non_first_level_class_inference_failed_fallback,
                                pre_class_inference_callback
                            )

                            type_variables_to_type_inference_results[type_variable] = top_inference_result

                        # Combine results for type variables.
                        type_inference_result = replace_type_variables_in_type_annotation(
                            top_class_prediction_subscribed_by_type_variables,
                            type_variables_to_type_inference_results
                        )

                    return_value = type_inference_result
                else:
                    return_value = top_class_prediction

                if can_be_none:
                    return_value = subscribe(TypeshedClass('typing', 'Optional'), (return_value,))

            self.type_inference_cache[equivalent_node_disjoint_set_top_nodes] = return_value

            return return_value
