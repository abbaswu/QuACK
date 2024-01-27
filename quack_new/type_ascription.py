import ast
import logging
import typing

from get_relation_sets_of_type_parameters import get_relation_sets_of_type_parameters
from relations import NonEquivalenceRelationType
from typeshed_client_ex.type_definitions import TypeshedTypeAnnotation, TypeshedTypeVariable, TypeshedClass, \
    Subscription, Union


async def type_ascription(
        node_set: frozenset[ast.AST],
        type_annotation: TypeshedTypeAnnotation,
        get_related_nodes_function: typing.Callable[
            [frozenset[ast.AST], NonEquivalenceRelationType, object], typing.Coroutine[None, None, frozenset[ast.AST]]],
        typeshed_type_variable_ascription_callback: typing.Callable[
            [frozenset[ast.AST], TypeshedTypeVariable, float], typing.Coroutine[typing.Any, typing.Any, None]],
        typeshed_class_ascription_callback: typing.Callable[
            [frozenset[ast.AST], TypeshedClass, float], typing.Coroutine[typing.Any, typing.Any, None]],
        weight: float = 1.0,
        visited_node_set: frozenset[ast.AST] = frozenset()
):
    nodes_to_ascribe_to = node_set - visited_node_set

    if not nodes_to_ascribe_to:
        return
    else:
        logging.info('Performing type ascription: %s <- %s, weight %s', nodes_to_ascribe_to, type_annotation, weight)
        if isinstance(type_annotation, TypeshedTypeVariable):
            await typeshed_type_variable_ascription_callback(nodes_to_ascribe_to, type_annotation, weight)
        elif isinstance(type_annotation, TypeshedClass):
            await typeshed_class_ascription_callback(nodes_to_ascribe_to, type_annotation, weight)
        elif isinstance(type_annotation, Subscription):
            new_visited_node_set = visited_node_set | node_set

            subscribed_class = type_annotation.subscribed_class
            type_annotation_tuple = type_annotation.type_annotation_tuple

            # Ascribe the subscribed class to the node set.
            await typeshed_class_ascription_callback(nodes_to_ascribe_to, type_annotation.subscribed_class, weight)

            # Handle the type annotation tuple based on the semantics of different subscribed classes.
            relation_sets_of_type_parameters = get_relation_sets_of_type_parameters(
                subscribed_class,
                len(type_annotation_tuple)
            )

            for type_annotation_in_type_annotation_tuple, relation_set in zip(type_annotation_tuple,
                                                                              relation_sets_of_type_parameters):
                related_nodes = set()
                for relation_type, parameter in relation_set:
                    related_nodes |= await get_related_nodes_function(nodes_to_ascribe_to, relation_type, parameter)

                await type_ascription(
                    frozenset(related_nodes),
                    type_annotation_in_type_annotation_tuple,
                    get_related_nodes_function,
                    typeshed_type_variable_ascription_callback,
                    typeshed_class_ascription_callback,
                    weight,
                    new_visited_node_set
                )
        elif isinstance(type_annotation, Union):
            for type_annotation_in_type_annotation_frozenset in type_annotation.type_annotation_frozenset:
                await type_ascription(
                    node_set,
                    type_annotation_in_type_annotation_frozenset,
                    get_related_nodes_function,
                    typeshed_type_variable_ascription_callback,
                    typeshed_class_ascription_callback,
                    weight / len(type_annotation.type_annotation_frozenset),
                    visited_node_set
                )
        else:
            logging.error('Cannot perform type ascription for type annotation %s!', type_annotation)
