import logging

from relations import NonEquivalenceRelationType, NonEquivalenceRelationAndParameter
from typeshed_client_ex.type_definitions import TypeshedClass


def get_relation_sets_of_type_parameters(
        subscribed_class: TypeshedClass,
        number_of_type_parameters: int = 0
) -> list[set[NonEquivalenceRelationAndParameter]]:
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
    # dict_values
    # Ascribe <the second type annotation> to nodes which are IterTargetOf the node set.
    elif subscribed_class == TypeshedClass('_collections_abc', 'dict_values'):
        return [{}, {(NonEquivalenceRelationType.IterTargetOf, None)}]
    # SupportsGetItem-like
    # Ascribe <the first type annotation> to nodes which are KeyOf the node set.
    # Ascribe <the second type annotation> to nodes which are ValueOf the node set.
    elif subscribed_class in (
            TypeshedClass('_typeshed', 'SupportsGetItem'),
            TypeshedClass('_typeshed', 'SupportsItemAccess'),
    ):
        return [{(NonEquivalenceRelationType.KeyOf, None)}, {(NonEquivalenceRelationType.ValueOf, None)}]
    # Sequence-like
    # Ascribe <the first type annotation> to nodes which are IterTargetOf, ValueOf the node set.
    elif subscribed_class in (
            TypeshedClass('typing', 'Sequence'),
            TypeshedClass('typing', 'MutableSequence'),
            TypeshedClass('builtins', 'list'),
            TypeshedClass('collections', 'deque'),
            TypeshedClass('collections', 'UserList'),
            TypeshedClass('array', 'array'),
            TypeshedClass('_typeshed', 'SupportsLenAndGetItem')
    ):
        return [{(NonEquivalenceRelationType.IterTargetOf, None), (NonEquivalenceRelationType.ValueOf, None)}]
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
        return [{(NonEquivalenceRelationType.IterTargetOf, None), (NonEquivalenceRelationType.KeyOf, None)},
                {(NonEquivalenceRelationType.ValueOf, None)}]
    # Counter-like
    # Ascribe <the first type annotation> to nodes which are IterTargetOf, KeyOf the node set.
    elif subscribed_class == TypeshedClass('collections', 'Counter'):
        return [{(NonEquivalenceRelationType.IterTargetOf, None), (NonEquivalenceRelationType.KeyOf, None)}]
    # Awaitable-like
    # Ascribe <the first type annotation> to nodes which are YieldFromAwaitResultOf the node set.
    elif subscribed_class == TypeshedClass('typing', 'Awaitable'):
        return [{(NonEquivalenceRelationType.YieldFromAwaitResultOf, None)}]
    # Generator-like
    # Ascribe <the first type annotation> to nodes which are IterTargetOf the node set.
    # Ascribe <the second type annotation> to nodes which are SendTargetOf the node set.
    # Ascribe <the third type annotation> to nodes which are YieldFromAwaitResultOf the node set.
    elif subscribed_class in (
            TypeshedClass('typing', 'Generator'),
            TypeshedClass('typing', 'Coroutine')
    ):
        return [{(NonEquivalenceRelationType.IterTargetOf, None)}, {(NonEquivalenceRelationType.SendTargetOf, None)},
                {(NonEquivalenceRelationType.YieldFromAwaitResultOf, None)}]
    # AsyncGenerator-like
    # Ascribe <the first type annotation> to nodes which are IterTargetOf the node set.
    # Ascribe <the second type annotation> to nodes which are SendTargetOf the node set.
    elif subscribed_class == TypeshedClass('typing', 'AsyncGenerator'):
        return [{(NonEquivalenceRelationType.IterTargetOf, None)}, {(NonEquivalenceRelationType.SendTargetOf, None)}]
    # tuple-like
    # Ascribe <the i-th type annotation> to nodes which are ElementOf i the node set.
    elif subscribed_class == TypeshedClass('builtins', 'tuple'):
        return [{(NonEquivalenceRelationType.ElementOf, i)} for i in range(number_of_type_parameters)]
    # Callable-like
    # Ascribe <the i-th type annotation> to nodes which are ArgumentOf i the node set.
    # Ascribe <the last type annotation> to nodes which are ReturnedValueOf the node set.
    elif subscribed_class in (
            TypeshedClass('typing', 'Callable'),
            TypeshedClass('builtins', 'staticmethod'),
            TypeshedClass('builtins', 'classmethod')
    ):
        return [{(NonEquivalenceRelationType.ArgumentOf, i)} for i in range(number_of_type_parameters - 1)] + [
            {(NonEquivalenceRelationType.ReturnedValueOf, None)}]
    else:
        logging.error('Unknown semantics of subscribed class %s!', subscribed_class)
        return []
