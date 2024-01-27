import ast
import logging
import typing

from relations import NonEquivalenceRelationType, NonEquivalenceRelationTuple, NonEquivalenceRelationGraph
from typeshed_client_ex.type_definitions import TypeshedTypeAnnotation, TypeshedTypeVariable, TypeshedClass, \
    Subscription, Union


def get_relation_sets_of_type_parameters(
        subscribed_class: TypeshedClass,
        number_of_type_parameters: int = 0
) -> list[set[tuple[NonEquivalenceRelationTuple]]]:
            # Iterable-like
            # Ascribe <the first type annotation> to nodes which are IterTargetOf the node set.
            if subscribed_class in (
                    TypeshedClass('typing', 'Iterable'),
                    TypeshedClass('typing', 'Iterator'),
                    TypeshedClass('typing', 'Container'),
                    TypeshedClass('typing', 'Collection'),
                    TypeshedClass('typing', 'AbstractSet'),
                    TypeshedClass('typing', 'MutableSet'),
                    TypeshedClass('typing', 'Reversible'),
                    TypeshedClass('typing', 'KeysView'),
                    TypeshedClass('typing', 'ValuesView'),
                    TypeshedClass('typing', 'AsyncIterable'),
                    TypeshedClass('typing', 'AsyncIterator'),
                    TypeshedClass('builtins', 'set'),
                    TypeshedClass('builtins', 'frozenset'),
                    TypeshedClass('_collections_abc', 'dict_keys'),
                    TypeshedClass('builtins', 'filter'),
                    TypeshedClass('builtins', 'map'),
                    TypeshedClass('builtins', 'reversed'),
                    TypeshedClass('builtins', 'zip'),
                    TypeshedClass('_typeshed', 'SupportsNext'),
                    TypeshedClass('_typeshed', 'SupportsAnext'),
                    TypeshedClass('itertools', 'count'),
                    TypeshedClass('itertools', 'cycle'),
                    TypeshedClass('itertools', 'repeat'),
                    TypeshedClass('itertools', 'accumulate'),
                    TypeshedClass('itertools', 'chain'),
                    TypeshedClass('itertools', 'compress'),
                    TypeshedClass('itertools', 'dropwhile'),
                    TypeshedClass('itertools', 'filterfalse'),
                    TypeshedClass('itertools', 'islice'),
                    TypeshedClass('itertools', 'starmap'),
                    TypeshedClass('itertools', 'takewhile'),
                    TypeshedClass('itertools', 'zip_longest'),
                    TypeshedClass('itertools', 'product'),
                    TypeshedClass('itertools', 'combinations'),
                    TypeshedClass('itertools', 'pairwise')
            ):
                return [{(NonEquivalenceRelationType.IterTargetOf, None)}]
            # ItemsView-like
            # Ascribe tuple[<the first type annotation>, <the second type annotation>] to nodes which are IterTargetOf the node set.
            elif subscribed_class in (
                    TypeshedClass('typing', 'ItemsView'),
                    TypeshedClass('_collections_abc', 'dict_items')
            ):
                if len(type_annotation_tuple) >= 2:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None),
                        Subscription(
                            TypeshedClass('builtins', 'tuple'),
                            (
                                type_annotation_tuple[0],
                                type_annotation_tuple[1]
                            )
                        ),
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # dict_values
            # Ascribe <the second type annotation> to nodes which are IterTargetOf the node set.
            elif subscribed_class == TypeshedClass('_collections_abc', 'dict_values'):
                if len(type_annotation_tuple) >= 2:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None),
                        type_annotation_tuple[1],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # enumerate-like
            # Ascribe tuple[int, <the first type annotation>] to nodes which are IterTargetOf the node set.
            elif subscribed_class == TypeshedClass('builtins', 'enumerate'):
                if len(type_annotation_tuple) >= 1:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None),
                        Subscription(
                            TypeshedClass('builtins', 'tuple'),
                            (
                                TypeshedClass('builtins', 'int'),
                                type_annotation_tuple[0]
                            )
                        ),
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # groupby-like
            # Ascribe tuple[<the first type annotation>, Iterator<the second type annotation>] to nodes which are IterTargetOf the node set.
            elif subscribed_class in (
                TypeshedClass('itertools', 'groupby'),
            ):
                if len(type_annotation_tuple) >= 2:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None),
                        Subscription(
                            TypeshedClass('builtins', 'tuple'),
                            (
                                type_annotation_tuple[0],
                                Subscription(
                                    TypeshedClass('typing', 'Iterator'),
                                    (
                                        type_annotation_tuple[1],
                                    )
                                )
                            )
                        ),
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # permutations-like
            # Ascribe Sequence[<the first type annotation>] to nodes which are IterTargetOf the node set.
            elif subscribed_class in (
                TypeshedClass('itertools', 'permutations'),
                TypeshedClass('itertools', 'combinations_with_replacement'),
            ):
                if len(type_annotation_tuple) >= 1:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None),
                        Subscription(
                            TypeshedClass('typing', 'Sequence'),
                            (
                                type_annotation_tuple[0],
                            )
                        ),
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # SupportsGetItem-like
            # Ascribe <the first type annotation> to nodes which are KeyOf the node set.
            # Ascribe <the second type annotation> to nodes which are ValueOf the node set.
            elif subscribed_class in (
                    TypeshedClass('_typeshed', 'SupportsGetItem'),
                    TypeshedClass('_typeshed', 'SupportsItemAccess'),
            ):
                if len(type_annotation_tuple) >= 1:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.KeyOf, None),
                        type_annotation_tuple[0],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )

                if len(type_annotation_tuple) >= 2:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.ValueOf, None),
                        type_annotation_tuple[1],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # Sequence-like
            # Ascribe <the first type annotation> to nodes which are IterTargetOf, ValueOf the node set.
            # Ascribe int to nodes which are KeyOf the node set.
            elif subscribed_class in (
                    TypeshedClass('typing', 'Sequence'),
                    TypeshedClass('typing', 'MutableSequence'),
                    TypeshedClass('builtins', 'list'),
                    TypeshedClass('collections', 'deque'),
                    TypeshedClass('collections', 'UserList'),
                    TypeshedClass('array', 'array'),
                    TypeshedClass('_typeshed', 'SupportsLenAndGetItem')
            ):
                if len(type_annotation_tuple) >= 1:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None) | await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.ValueOf, None),
                        type_annotation_tuple[0],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )

                await type_ascription(
                    await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.KeyOf, None),
                    TypeshedClass('builtins', 'int'),
                    get_related_nodes_function,
                    typeshed_type_variable_ascription_callback,
                    typeshed_class_ascription_callback,
                    weight,
                    new_visited_node_set
                )
            # Mapping-like
            # Ascribe <the first type annotation> to nodes which are IterTargetOf, KeyOf the node set.
            # Ascribe <the second type annotation> to nodes which are ValueOf the node set.
            elif subscribed_class in (
                    TypeshedClass('_typeshed', 'SupportsKeysAndGetItem'),
                    TypeshedClass('_typeshed', 'SupportsItems'),
                    TypeshedClass('typing', 'Mapping'),
                    TypeshedClass('typing', 'MutableMapping'),
                    TypeshedClass('types', 'MappingProxyType'),
                    TypeshedClass('builtins', 'dict'),
                    TypeshedClass('collections', 'ChainMap'),
                    TypeshedClass('collections', 'defaultdict'),
                    TypeshedClass('collections', 'OrderedDict')
            ):
                if len(type_annotation_tuple) >= 1:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None) | await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.KeyOf, None),
                        type_annotation_tuple[0],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )

                if len(type_annotation_tuple) >= 2:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.ValueOf, None),
                        type_annotation_tuple[1],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # Counter-like
            # Ascribe <the first type annotation> to nodes which are IterTargetOf, KeyOf the node set.
            # Ascribe int to nodes which are ValueOf the node set.
            elif subscribed_class == TypeshedClass('collections', 'Counter'):
                if len(type_annotation_tuple) >= 1:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None) | await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.KeyOf, None),
                        type_annotation_tuple[0],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )

                await type_ascription(
                    await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.ValueOf, None),
                    TypeshedClass('builtins', 'int'),
                    get_related_nodes_function,
                    typeshed_type_variable_ascription_callback,
                    typeshed_class_ascription_callback,
                    weight,
                    new_visited_node_set
                )
            # Awaitable-like
            # Ascribe <the first type annotation> to nodes which are YieldFromAwaitResultOf the node set.
            elif subscribed_class == TypeshedClass('typing', 'Awaitable'):
                if len(type_annotation_tuple) >= 1:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.YieldFromAwaitResultOf, None),
                        type_annotation_tuple[0],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # Generator-like
            # Ascribe <the first type annotation> to nodes which are IterTargetOf the node set.
            # Ascribe <the second type annotation> to nodes which are SendTargetOf the node set.
            # Ascribe <the third type annotation> to nodes which are YieldFromAwaitResultOf the node set.
            elif subscribed_class in (
                    TypeshedClass('typing', 'Generator'),
                    TypeshedClass('typing', 'Coroutine')
            ):
                if len(type_annotation_tuple) >= 1:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None),
                        type_annotation_tuple[0],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )

                if len(type_annotation_tuple) >= 2:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.SendTargetOf, None),
                        type_annotation_tuple[1],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )

                if len(type_annotation_tuple) >= 3:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.YieldFromAwaitResultOf, None),
                        type_annotation_tuple[2],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # AsyncGenerator-like
            # Ascribe <the first type annotation> to nodes which are IterTargetOf the node set.
            # Ascribe <the second type annotation> to nodes which are SendTargetOf the node set.
            elif subscribed_class == TypeshedClass('typing', 'AsyncGenerator'):
                if len(type_annotation_tuple) >= 1:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.IterTargetOf, None),
                        type_annotation_tuple[0],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )

                if len(type_annotation_tuple) >= 2:
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.SendTargetOf, None),
                        type_annotation_tuple[1],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            # tuple-like
            # Ascribe <the i-th type annotation> to nodes which are ElementOf i the node set.
            # Ascribe int to nodes which are KeyOf the node set.
            elif subscribed_class == TypeshedClass('builtins', 'tuple'):
                for i, element_type_annotation in enumerate(type_annotation_tuple):
                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.ElementOf, i),
                        element_type_annotation,
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )

                await type_ascription(
                    await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.KeyOf, None),
                    TypeshedClass('builtins', 'int'),
                    get_related_nodes_function,
                    typeshed_type_variable_ascription_callback,
                    typeshed_class_ascription_callback,
                    weight,
                    new_visited_node_set
                )
            # Callable-like
            # Ascribe <the i-th type annotation> to nodes which are ArgumentOf i the node set.
            # Ascribe <the last type annotation> to nodes which are ReturnedValueOf the node set.
            elif subscribed_class in (
                    TypeshedClass('typing', 'Callable'),
                    TypeshedClass('builtins', 'staticmethod'),
                    TypeshedClass('builtins', 'classmethod')
            ):
                if len(type_annotation_tuple) >= 1:
                    for i, argument_type_annotation in enumerate(type_annotation_tuple[:-1]):
                        await type_ascription(
                            await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.ArgumentOf, i),
                            argument_type_annotation,
                            get_related_nodes_function,
                            typeshed_type_variable_ascription_callback,
                            typeshed_class_ascription_callback,
                            weight,
                            new_visited_node_set
                        )

                    await type_ascription(
                        await get_related_nodes_function(nodes_to_ascribe_to, NonEquivalenceRelationType.ReturnedValueOf, None),
                        type_annotation_tuple[-1],
                        get_related_nodes_function,
                        typeshed_type_variable_ascription_callback,
                        typeshed_class_ascription_callback,
                        weight,
                        new_visited_node_set
                    )
            else:
                logging.error('Unknown semantics of subscribed class %s!', subscribed_class)
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
