import _ast
import ast
import json
import logging
import typing
from collections import defaultdict, Counter

import numpy as np

import switches_singleton
import typing_constraints_singleton
from breadth_first_search_layers import breadth_first_search_layers
from class_query_database import ClassQueryDatabase
from determine_number_of_type_variables import determine_number_of_type_variables
from get_attributes_in_runtime_class import get_attributes_in_runtime_class
from get_relation_sets_of_type_parameters import get_relation_sets_of_type_parameters
from relations import NonEquivalenceRelationType
from runtime_term import Instance
from type_definitions import RuntimeClass
from typeshed_client_ex.client import Client
from typeshed_client_ex.type_definitions import TypeshedTypeAnnotation, TypeshedClass, from_runtime_class, \
    subscribe


def dump_confidence_and_possible_class_list(
        confidence_and_possible_class_list: list[tuple[float, TypeshedClass]],
        class_inference_log_file_io: typing.IO
):
    confidence_and_possible_class_string_list_list: list[list[float | str]] = [
        [confidence, str(possible_class)]
        for confidence, possible_class in confidence_and_possible_class_list
    ]

    json.dump(confidence_and_possible_class_string_list_list, class_inference_log_file_io)
    class_inference_log_file_io.write('\n')


def get_all_related_nodes(augmented_node_set, relation_set):
    all_related_nodes = set()
    for node in augmented_node_set:
        for relation_type, parameter in relation_set:
            related_nodes = typing_constraints_singleton.node_non_equivalence_relation_graph.get_out_nodes_with_relation_type_and_parameter(
                node, relation_type, parameter)
            all_related_nodes.update(related_nodes)

    return all_related_nodes


def query_relation_sets_of_type_parameters(top_class_prediction, augmented_node_set):
    relation_type_to_parameter_to_out_nodes: defaultdict[
        NonEquivalenceRelationType,
        defaultdict[typing.Optional[typing.Any], set[_ast.AST]]
    ] = defaultdict(lambda: defaultdict(set))

    for node in augmented_node_set:
        for relation_type, parameter_to_out_nodes in typing_constraints_singleton.node_non_equivalence_relation_graph.get_out_nodes(
                node).items():
            for parameter, out_nodes in parameter_to_out_nodes.items():
                relation_type_to_parameter_to_out_nodes[relation_type][parameter].update(out_nodes)

    number_of_type_variables = determine_number_of_type_variables(
        top_class_prediction,
        relation_type_to_parameter_to_out_nodes,
        typing_constraints_singleton.client
    )

    return get_relation_sets_of_type_parameters(
        top_class_prediction,
        number_of_type_variables
    )


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
            not_none_instance_classes: set[RuntimeClass] = set()

            runtime_term_sharing_node_disjoint_set_top_node_set = {
                typing_constraints_singleton.runtime_term_sharing_node_disjoint_set.find(node)
                for node in nodes
            }

            for runtime_term_sharing_node_disjoint_set_top_node in runtime_term_sharing_node_disjoint_set_top_node_set:
                runtime_term_set = typing_constraints_singleton.runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets[
                    runtime_term_sharing_node_disjoint_set_top_node
                ]
                for runtime_term in runtime_term_set:
                    if isinstance(runtime_term, Instance):
                        instance_class = runtime_term.class_
                        if instance_class is type(None):
                            can_be_none = True
                        elif instance_class is not type(NotImplemented):
                            not_none_instance_classes.add(instance_class)

            logging.info(
                '%sCan %s be None? %s',
                indent,
                nodes, can_be_none
            )

            # Initialize aggregate attribute counter.

            aggregate_attribute_counter: Counter[str] = Counter()
            for node in nodes:
                attribute_counter = typing_constraints_singleton.nodes_to_attribute_counters[node]
                aggregate_attribute_counter.update(attribute_counter)

            logging.info(
                '%sAggregate attribute counter for %s: %s',
                indent,
                nodes, aggregate_attribute_counter
            )

            # Query possible classes.

            confidence_and_possible_class_list: list[tuple[float, TypeshedClass]] = list()

            if (
                    switches_singleton.shortcut_single_class_covering_all_attributes
                    and len(not_none_instance_classes) == 1
                    and set(aggregate_attribute_counter.keys()).issubset(
                get_attributes_in_runtime_class(next(iter(not_none_instance_classes)))
            )
            ):

                single_instance_class_covering_all_attributes = next(iter(not_none_instance_classes))
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
                # Part 0: Augment node set.
                augmented_node_set = set()
                for breadth_first_search_layer in breadth_first_search_layers(
                        typing_constraints_singleton.node_containment_graph,
                        node_set
                ):
                    augmented_node_set.update(breadth_first_search_layer)

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
                    frozenset(augmented_node_set),
                    depth + 1,
                    cosine_similarity_threshold
                )

                if class_inference_log_file_io is not None:
                    dump_confidence_and_possible_class_list(
                        confidence_and_possible_class_list,
                        class_inference_log_file_io
                    )

                # Part 2: Extract top class prediction.

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

                # Part 3: Predict type parameters of top class prediction.

                type_parameter_type_prediction_list = []

                for relation_set in query_relation_sets_of_type_parameters(top_class_prediction, augmented_node_set):
                    related_nodes = get_all_related_nodes(augmented_node_set, relation_set)
                    type_parameter_type_prediction = self.infer_type(
                        frozenset(related_nodes),
                        depth + 1,
                        cosine_similarity_threshold,
                        depth_limit,
                        first_level_class_inference_failed_fallback,
                        non_first_level_class_inference_failed_fallback,
                        class_inference_log_file_io
                    )
                    type_parameter_type_prediction_list.append(type_parameter_type_prediction)

                # Part 4: Get final type prediction.
                if type_parameter_type_prediction_list and switches_singleton.predict_type_parameters:
                    final_type_prediction = subscribe(top_class_prediction, tuple(type_parameter_type_prediction_list))
                else:
                    final_type_prediction = top_class_prediction

                return_value = final_type_prediction

            self.type_inference_cache[node_set] = return_value

            return return_value
