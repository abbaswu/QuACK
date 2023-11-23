import builtins
import collections.abc
import logging
import numbers
import typing
import types

from collections import Counter
from math import log
from types import ModuleType

from numpy import ndarray, zeros, fromiter
from scipy.spatial.distance import cosine

import switches_singleton
from get_attributes_in_runtime_class import get_attributes_in_runtime_class
from get_types_in_module import get_types_in_module
from inheritance_graph import construct_inheritance_graph, iterate_inheritance_graph
from set_trie import SetTrieNode, add, contains
from typeshed_client_ex.client import Client
from typeshed_client_ex.type_definitions import TypeshedClass, from_runtime_class, get_attributes_in_class_definition


class ClassQueryDatabase:
    def __init__(
            self,
            module_set: typing.AbstractSet[ModuleType],
            typeshed_client: Client
    ):
        self.candidate_class_list: list[TypeshedClass] = []
        self.candidate_class_to_attribute_set_dict: dict[TypeshedClass, set[str]] = dict()

        self.candidate_class_to_index_dict: dict[TypeshedClass, int] = dict()
        self.number_of_types: int = 0

        self.attribute_frequency_counter: Counter[str] = Counter()
        self.attribute_list: list[str] = []
        self.attribute_to_index_dict: dict[str, int] = dict()
        self.attribute_to_idf_dict: dict[str, float] = dict()
        self.number_of_attributes: int = 0

        self.candidate_class_to_idf_ndarray_dict: dict[TypeshedClass, ndarray] = dict()

        # Perform live object lookup for classes

        typeshed_class_to_attribute_set_dict: dict[TypeshedClass, set[str]] = dict()

        attribute_set_trie_root: SetTrieNode[str] = SetTrieNode()

        for typeshed_class in (
                TypeshedClass('_typeshed', 'SupportsItemAccess'),
                TypeshedClass('_typeshed', 'SupportsGetItem'),
                TypeshedClass('_typeshed', 'HasFileno'),
                TypeshedClass('_typeshed', 'SupportsRead'),
                TypeshedClass('_typeshed', 'SupportsReadline'),
                TypeshedClass('_typeshed', 'SupportsNoArgReadline'),
                TypeshedClass('_typeshed', 'SupportsWrite'),
                TypeshedClass('_typeshed', 'SupportsAdd'),
                TypeshedClass('_typeshed', 'SupportsRAdd'),
                TypeshedClass('_typeshed', 'SupportsSub'),
                TypeshedClass('_typeshed', 'SupportsRSub'),
                TypeshedClass('_typeshed', 'SupportsDivMod'),
                TypeshedClass('_typeshed', 'SupportsRDivMod'),
                TypeshedClass('_typeshed', 'SupportsTrunc'),
                TypeshedClass('_typeshed', 'structseq'),
        ):
            typeshed_class_definition = typeshed_client.get_class_definition(typeshed_class)
            attributes_in_typeshed_class = get_attributes_in_class_definition(typeshed_class_definition) | get_attributes_in_runtime_class(object)
            add(attribute_set_trie_root, attributes_in_typeshed_class)
            typeshed_class_to_attribute_set_dict[typeshed_class] = attributes_in_typeshed_class

        attribute_frozenset_to_candidate_class_set_dict: collections.defaultdict[frozenset[str], set[type]] = collections.defaultdict(set)

        for runtime_class in (
                object,
                int,
                float,
                complex,
                list,
                bytearray,
                memoryview,
                bytes,
                str,
                set,
                frozenset,
                dict,
                tuple,
                range,
                slice,
                type,
                BaseException,
                OSError,
                SyntaxError,
                NameError,
                ImportError,
                AttributeError,
                SystemExit,
                StopIteration,
                numbers.Complex,
                numbers.Real,
                numbers.Rational,
                numbers.Integral,
                collections.abc.Iterable,
                collections.abc.Collection,
                collections.abc.Iterator,
                collections.abc.Reversible,
                collections.abc.Generator,
                collections.abc.AsyncIterable,
                collections.abc.AsyncIterator,
                collections.abc.AsyncGenerator,
                collections.abc.Awaitable,
                collections.abc.Coroutine,
                collections.abc.Sequence,
                collections.abc.MutableSequence,
                collections.abc.Mapping,
                collections.abc.MutableMapping,
                collections.abc.Set,
                collections.abc.MutableSet,
                collections.abc.Callable,
                typing.SupportsIndex,
                typing.SupportsBytes,
                typing.SupportsComplex,
                typing.SupportsFloat,
                typing.SupportsInt,
                typing.SupportsRound,
                typing.SupportsAbs,
                typing.TextIO,
                typing.IO,
                types.CellType,
                types.TracebackType,
                types.FrameType,
                types.CodeType,
        ):
            attributes_in_runtime_class = get_attributes_in_runtime_class(runtime_class)
            add(attribute_set_trie_root, attributes_in_runtime_class)
            attribute_frozenset_to_candidate_class_set_dict[frozenset(attributes_in_runtime_class)].add(runtime_class)

            typeshed_class = from_runtime_class(runtime_class)
            typeshed_class_to_attribute_set_dict[typeshed_class] = attributes_in_runtime_class

        runtime_class_set_from_modules: set[type] = set()

        excluded_module_names = (
            'builtins',
            'types',
            'typing',
            'typing_extensions',
            'collections.abc',
            'numbers'
        )

        for module in module_set:
            if module.__name__ not in excluded_module_names:
                for runtime_class in get_types_in_module(module):
                    for mro_entry in runtime_class.__mro__:
                        if mro_entry.__module__ not in excluded_module_names:
                            runtime_class_set_from_modules.add(mro_entry)

        inheritance_graph = construct_inheritance_graph(runtime_class_set_from_modules)
        for runtime_class in iterate_inheritance_graph(inheritance_graph):
            attributes_in_runtime_class = get_attributes_in_runtime_class(runtime_class)
            if contains(attribute_set_trie_root, attributes_in_runtime_class):
                type_set = attribute_frozenset_to_candidate_class_set_dict[frozenset(attributes_in_runtime_class)]
                try:
                    if issubclass(runtime_class, tuple(type_set)):
                        logging.warning('Excluded runtime class %s from class query database as a superclass in %s covers its attribute set', runtime_class, type_set)
                        continue
                except TypeError:
                    logging.exception('Excluded runtime class %s from class query database.', runtime_class)
                    continue

            add(attribute_set_trie_root, attributes_in_runtime_class)
            attribute_frozenset_to_candidate_class_set_dict[frozenset(attributes_in_runtime_class)].add(runtime_class)

            typeshed_class = from_runtime_class(runtime_class)
            typeshed_class_to_attribute_set_dict[typeshed_class] = attributes_in_runtime_class

        for typeshed_class, attribute_set in typeshed_class_to_attribute_set_dict.items():
            self.candidate_class_list.append(typeshed_class)
            self.candidate_class_to_attribute_set_dict[typeshed_class] = attribute_set

        for index, typeshed_class in enumerate(self.candidate_class_list):
            self.candidate_class_to_index_dict[typeshed_class] = index

        self.number_of_types += len(self.candidate_class_list)

        for typeshed_class, attribute_set in self.candidate_class_to_attribute_set_dict.items():
            for attribute in attribute_set:
                self.attribute_frequency_counter[attribute] += 1

        self.attribute_list.extend(self.attribute_frequency_counter.keys())

        for index, attribute in enumerate(self.attribute_list):
            self.attribute_to_index_dict[attribute] = index

        for attribute, attribute_frequency in self.attribute_frequency_counter.items():
            # https://towardsdatascience.com/how-sklearns-tf-idf-is-different-from-the-standard-tf-idf-275fa582e73d
            self.attribute_to_idf_dict[attribute] = log((self.number_of_types) / (1 + attribute_frequency))

        self.number_of_attributes += len(self.attribute_list)

        for typeshed_class, attribute_set in self.candidate_class_to_attribute_set_dict.items():
            idf_ndarray, _ = self.get_idf_ndarray(attribute_set)
            self.candidate_class_to_idf_ndarray_dict[typeshed_class] = idf_ndarray

    def get_idf_ndarray(self, attribute_set: set[str]) -> tuple[ndarray, int]:
        idf_ndarray = zeros(self.number_of_attributes)
        number_of_attributes: int = 0
        for attribute in attribute_set:
            try:
                index = self.attribute_to_index_dict[attribute]
                idf = self.attribute_to_idf_dict[attribute]
                idf_ndarray[index] = idf
                number_of_attributes += 1
            except KeyError:
                logging.warning('Skipped attribute: %s', attribute)
        return idf_ndarray, number_of_attributes

    def get_tf_idf_ndarray(self, attribute_counter: Counter[str]) -> tuple[ndarray, int]:
        tf_idf_ndarray = zeros(self.number_of_attributes)
        number_of_attributes: int = 0
        for attribute, count in attribute_counter.items():
            tf = count
            try:
                index = self.attribute_to_index_dict[attribute]
                idf = self.attribute_to_idf_dict[attribute]
                tf_idf_ndarray[index] = tf * idf
                number_of_attributes += 1
            except KeyError:
                logging.warning('Skipped attribute: %s', attribute)
        return tf_idf_ndarray, number_of_attributes

    # Query TypeshedClass's given an attribute set.
    def query(self, attribute_counter: Counter[str]) -> tuple[ndarray, ndarray]:
        query_tf_idf_ndarray, number_of_attributes = self.get_tf_idf_ndarray(attribute_counter)

        if number_of_attributes > 0:
            class_ndarray: ndarray = fromiter(
                self.candidate_class_to_idf_ndarray_dict.keys(),
                object,
                len(self.candidate_class_to_idf_ndarray_dict.keys())
            )

            cosine_similarity_ndarray = zeros(len(class_ndarray))

            for i, (candidate_class, idf_ndarray) in enumerate(
                    self.candidate_class_to_idf_ndarray_dict.items()
            ):
                cosine_distance: float = cosine(
                    query_tf_idf_ndarray,
                    idf_ndarray
                )

                cosine_similarity = 1 - cosine_distance

                cosine_similarity_ndarray[i] = cosine_similarity
        else:
            class_ndarray = zeros(0, dtype=object)
            cosine_similarity_ndarray = zeros(0)

        return class_ndarray, cosine_similarity_ndarray
