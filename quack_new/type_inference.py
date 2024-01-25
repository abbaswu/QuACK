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

import node_textual_representation_singleton
import switches_singleton
import typing_constraints_singleton
from breadth_first_search_layers import breadth_first_search_layers
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
            type_query_database: ClassQueryDatabase,
            client: Client
    ):
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

    def infer_classes_for_nodes(
            self,
            nodes: frozenset[ast.AST],
            indent_level: int = 0,
            cosine_similarity_threshold: float = 1e-1
    ) -> tuple[
        list[tuple[float, TypeshedClass]],  # class inference confidences and classs
        bool  # whether runtime class can be instance-of types.NoneType
    ]:
        indent = '    ' * indent_level

        # Has a record in cache
        if nodes in self.class_inference_cache:
            logging.info(
                '%sCache hit when performing class inference for %s.',
                indent,
                nodes
            )

            return self.class_inference_cache[nodes]
        else:
            # No record in cache
            logging.info(
                '%sCache miss when performing class inference for %s.',
                indent,
                nodes
            )

            # Determine whether it can be None.

            can_be_none: bool = False
            non_none_instance_classes: set[RuntimeClass] = set()

            runtime_term_sharing_node_disjoint_set_top_node_set = {
                typing_constraints_singleton.runtime_term_sharing_node_disjoint_set.find(node)
                for node in nodes
            }

            for runtime_term_sharing_node_disjoint_set_top_node in runtime_term_sharing_node_disjoint_set_top_node_set:
                runtime_term_set = \
                typing_constraints_singleton.runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets[
                    runtime_term_sharing_node_disjoint_set_top_node
                ]
                for runtime_term in runtime_term_set:
                    if isinstance(runtime_term, Instance):
                        instance_class = runtime_term.class_
                        if isinstance(instance_class, type(None)):
                            can_be_none = True
                        else:
                            non_none_instance_classes.add(instance_class)

            logging.info(
                '%sCan %s be None? %s',
                indent,
                nodes, can_be_none
            )

            # Initialize aggregate attribute counter.
            aggregate_attribute_counter: Counter[str] = Counter()

            for breadth_first_search_layer in breadth_first_search_layers(
                    typing_constraints_singleton.node_containment_graph,
                    nodes
            ):
                logging.info(
                    '%sCurrent breadth-first-search layer: %s', indent, breadth_first_search_layer
                )

                for node in breadth_first_search_layer:
                    attribute_counter = typing_constraints_singleton.nodes_to_attribute_counters[
                        node
                    ]

                    logging.info(
                        '%sAttribute counter for %s: %s',
                        indent,
                        node_textual_representation_singleton.node_to_textual_representation_dict.get(
                            node,
                            str(node)
                        ),
                        attribute_counter
                    )

                    aggregate_attribute_counter.update(attribute_counter)

            logging.info(
                '%sAggregate attribute counter for %s: %s',
                indent,
                nodes,
                aggregate_attribute_counter
            )

            # Query possible classes.

            confidence_and_possible_class_list: list[tuple[float, TypeshedClass]] = list()

            if (
                    switches_singleton.shortcut_single_class_covering_all_attributes
                    and len(non_none_instance_classes) == 1
                    and set(aggregate_attribute_counter.keys()).issubset(
                        get_attributes_in_runtime_class(next(iter(non_none_instance_classes)))
                    )
            ):

                single_instance_class_covering_all_attributes = next(iter(non_none_instance_classes))
                confidence_and_possible_class_list.append(
                    (1, from_runtime_class(single_instance_class_covering_all_attributes))
                )
            else:
                (
                    possible_class_ndarray,
                    cosine_similarity_ndarray,
                ) = self.type_query_database.query(aggregate_attribute_counter)

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
                    nodes,
                    confidence_and_possible_class_list
                )

            return_value = confidence_and_possible_class_list, can_be_none

            self.class_inference_cache[nodes] = return_value

            return return_value

    def infer_type(
            self,
            node_set: frozenset[ast.AST],
            depth: int = 0,
            cosine_similarity_threshold: float = 1e-1,
            depth_limit: int = 3,
            first_level_class_inference_failed_fallback: TypeshedClass = TypeshedClass('typing', 'Any'),
            non_first_level_class_inference_failed_fallback: TypeshedClass = TypeshedClass('typing', 'Any'),
            class_inference_log_file_io: typing.Optional[typing.IO] = None
    ) -> TypeshedTypeAnnotation:
        indent = '    ' * depth

        # Has a record in cache
        if node_set in self.type_inference_cache:
            logging.info(
                '%sCache hit when performing type inference for %s.',
                indent,
                node_set
            )

            return self.type_inference_cache[node_set]
        else:
            # No record in cache
            logging.info(
                '%sCache miss when performing type inference for %s.',
                indent,
                node_set
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
                    node_set
                )

                (
                    confidence_and_possible_class_list,
                    can_be_none
                ) = self.infer_classes_for_nodes(
                    node_set,
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
                        node_set,
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

                return_value = top_class_prediction

            self.type_inference_cache[node_set] = return_value

            return return_value
