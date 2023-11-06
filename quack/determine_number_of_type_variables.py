import ast
import logging
import typing
from collections import defaultdict

from relations import NonEquivalenceRelationGraph, NonEquivalenceRelationTuple, NonEquivalenceRelationType
from runtime_term import RuntimeTerm, Instance
from typeshed_client_ex.client import Client
from typeshed_client_ex.type_definitions import TypeshedClass, TypeshedClassDefinition


def determine_number_of_type_variables(
        node_set: typing.AbstractSet[ast.AST],
        typeshed_class: TypeshedClass,
        non_equivalence_relation_graph: NonEquivalenceRelationGraph,
        client: Client
) -> int:
    # Special handling for builtins.tuple and typing.Callable
    if typeshed_class in (
            TypeshedClass('builtins', 'tuple'),
            TypeshedClass('typing', 'Callable')
    ):
        all_out_edges_by_relation_tuple: defaultdict[NonEquivalenceRelationTuple, set[ast.AST]] = defaultdict(set)

        for node in node_set:
            out_edges_by_relation_tuple = non_equivalence_relation_graph.get_out_edges_by_relation_tuple(node)
            for relation_tuple, out_edges in out_edges_by_relation_tuple.items():
                all_out_edges_by_relation_tuple[relation_tuple].update(out_edges)

        # For builtins.tuple,
        # Determine the number of tuple elements.
        if typeshed_class == TypeshedClass('builtins', 'tuple'):
            element_index_set: set[int] = set()

            for relation_tuple, out_edges in all_out_edges_by_relation_tuple.items():
                relation_type, *parameters = relation_tuple
                if relation_type == NonEquivalenceRelationType.ElementOf:
                    element_index: int = parameters[0]
                    element_index_set.add(element_index)

            if element_index_set:
                return max(element_index_set) + 1
            else:
                return 0
        # For typing.Callable,
        # Determine the number of apparent arguments.
        elif typeshed_class == TypeshedClass('typing', 'Callable'):
            apparent_argument_index_set: set[int] = set()
            returned_value_of_relation_found: bool = False

            for relation_tuple, out_edges in all_out_edges_by_relation_tuple.items():
                relation_type, *parameters = relation_tuple
                if relation_type == NonEquivalenceRelationType.ArgumentOf:
                    apparent_argument_index: int = parameters[0]
                    apparent_argument_index_set.add(apparent_argument_index)
                elif relation_type == NonEquivalenceRelationType.ReturnedValueOf:
                    returned_value_of_relation_found = True

            if apparent_argument_index_set:
                number_of_apparent_arguments: int = max(apparent_argument_index_set) + 1
                # Add 1 for the return value.
                return number_of_apparent_arguments + 1
            else:
                if returned_value_of_relation_found:
                    return 1
                else:
                    return 0
        else:
            return 0
    # Handle other cases.
    else:
        try:
            class_definition: TypeshedClassDefinition = client.get_class_definition(typeshed_class)
            return len(class_definition.type_variable_list)
        except:
            logging.exception(f'Failed to get class definition for %s from Typeshed.', typeshed_class)
            return 0
