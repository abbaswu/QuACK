import builtins
import collections.abc
import contextlib
import io
import logging
import math
import numbers
import types
import typing
import typing_extensions

from collections import Counter
from math import log
from types import ModuleType

import numpy as np
import pandas as pd
from numpy import ndarray, zeros, fromiter
from scipy.spatial.distance import cosine

import switches_singleton
from breadth_first_search_layers import breadth_first_search_layers
from get_attributes_in_runtime_class import get_attributes_in_runtime_class
from get_types_in_module import get_types_in_module
from inheritance_graph import construct_base_to_derived_graph
from set_trie import SetTrieNode, add, contains
from typeshed_client_ex.client import Client
from typeshed_client_ex.type_definitions import TypeshedClass, from_runtime_class, get_attributes_in_class_definition


class ClassQueryDatabase:
    def __init__(
            self,
            module_set: typing.AbstractSet[ModuleType],
            typeshed_client: Client
    ):
        self.candidate_class_list: list[TypeshedClass]
        self.candidate_class_to_attribute_set_dict: dict[TypeshedClass, set[str]]
        self.candidate_class_to_index_dict: dict[TypeshedClass, int]
        self.candidate_class_ndarray: np.ndarray
        self.num_classes: int

        self.attribute_list: list[str]
        self.attribute_to_index_dict: dict[str, int]
        self.num_attributes: int

        # Document-term matrix
        self.class_attribute_matrix: pd.DataFrame
        # Count non-zero values in each column (Document Frequency)
        self.doc_frequency: pd.Series
        # Calculate IDF for each attribute
        self.idf: pd.Series
        # Calculate the average number of attributes in all classes
        self.average_num_attributes_in_classes: float

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

        # Special handling for byte sequences
        bytestring_typeshed_class = TypeshedClass('typing', 'ByteString')
        bytestring_attributes = get_attributes_in_runtime_class(bytes) | get_attributes_in_runtime_class(bytearray) | get_attributes_in_runtime_class(memoryview)
        add(attribute_set_trie_root, bytestring_attributes)
        typeshed_class_to_attribute_set_dict[bytestring_typeshed_class] = bytestring_attributes

        for runtime_class in (
                object,
                int,
                float,
                complex,
                list,
                str,
                set,
                frozenset,
                dict,
                tuple,
                range,
                slice,
                type,
                types.CellType,
                types.TracebackType,
                types.FrameType,
                types.CodeType,
                typing.SupportsIndex,
                typing.SupportsBytes,
                typing.SupportsComplex,
                typing.SupportsFloat,
                typing.SupportsInt,
                typing.SupportsRound,
                typing.SupportsAbs,
                typing.TextIO,
                typing.IO,
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
                numbers.Complex,
                numbers.Real,
                numbers.Rational,
                numbers.Integral,
                contextlib.AbstractContextManager,
                contextlib.AbstractAsyncContextManager
        ):
            attributes_in_runtime_class = get_attributes_in_runtime_class(runtime_class)
            add(attribute_set_trie_root, attributes_in_runtime_class)
            attribute_frozenset_to_candidate_class_set_dict[frozenset(attributes_in_runtime_class)].add(runtime_class)

            typeshed_class = from_runtime_class(runtime_class)
            typeshed_class_to_attribute_set_dict[typeshed_class] = attributes_in_runtime_class

        runtime_class_set_from_modules: set[type] = set()

        for module in module_set - {builtins, types, typing, typing_extensions, io, collections.abc, numbers, contextlib}: # Exclude some modules
            for runtime_class in get_types_in_module(module):
                runtime_class_set_from_modules.update(runtime_class.__mro__)

        base_to_derived_graph = construct_base_to_derived_graph(runtime_class_set_from_modules)

        for runtime_class_set in breadth_first_search_layers(base_to_derived_graph):
            for runtime_class in runtime_class_set:
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

        # Finish adding all classes
        # Initialize class query

        self.candidate_class_list = []
        self.candidate_class_to_attribute_set_dict = {}

        for typeshed_class, attribute_set in typeshed_class_to_attribute_set_dict.items():
            self.candidate_class_list.append(typeshed_class)
            self.candidate_class_to_attribute_set_dict[typeshed_class] = attribute_set

        self.candidate_class_to_index_dict = {}

        for index, typeshed_class in enumerate(self.candidate_class_list):
            self.candidate_class_to_index_dict[typeshed_class] = index
        
        self.candidate_class_ndarray = np.array(self.candidate_class_list)

        self.num_classes = len(self.candidate_class_list)

        attribute_set = set.union(*self.candidate_class_to_attribute_set_dict.values())

        self.attribute_list = list(attribute_set)

        self.attribute_to_index_dict = {
            attribute: index
            for index, attribute in enumerate(self.attribute_list)
        }

        self.num_attributes = len(self.attribute_list)

        # Document-term matrix
        self.class_attribute_matrix = pd.DataFrame(
            [
                [
                    (attribute in self.candidate_class_to_attribute_set_dict.get(candidate_class, set()))
                    for attribute in self.attribute_list
                ]
                for candidate_class in self.candidate_class_list
            ],
            index=list(map(str, self.candidate_class_list)),
            columns=self.attribute_list
        )

        # Count non-zero values in each column (Document Frequency)
        self.doc_frequency = self.class_attribute_matrix.apply(
            lambda attribute_column: (attribute_column != 0).sum()
        )

        # Calculate IDF for each attribute
        self.idf = np.log((self.num_classes - self.doc_frequency + 0.5) / (self.doc_frequency + 0.5) + 1)

        # Calculate the average number of attributes in all classes
        self.average_num_attributes_in_classes = (self.class_attribute_matrix != 0).sum(axis=1).mean()
    
    def get_score_function(self, attributes: typing.Iterable[str]):
        k_1 = 1.5
        b = 0.75

        def score_function(class_row: pd.Series):
            num_attributes_in_class = (class_row != 0).sum()
            
            def attribute_score(attribute: str):
                attribute_idf = self.idf.get(attribute, 0)
                attribute_frequency = class_row.get(attribute, 0)
                return attribute_idf * (attribute_frequency * (k_1 + 1)) / (attribute_frequency + k_1 * (1 - b + b * (num_attributes_in_class) / self.average_num_attributes_in_classes))

            return sum(
                map(attribute_score, attributes),
                0
            )
        
        return score_function

    # Query TypeshedClass's given an attribute set.
    def query(self, attribute_counter: Counter[str]) -> tuple[ndarray, ndarray]:
        if attribute_counter:
            score_function = self.get_score_function(attribute_counter)
            result_series = self.class_attribute_matrix.apply(score_function, axis=1)

            # Use numpy.argsort() on the Series values
            indices = np.argsort(result_series.values)[::-1]

            class_ndarray = self.candidate_class_ndarray[indices]
            similarity_ndarray = result_series.values[indices]

            if (max_similarity := similarity_ndarray[0]) > 0.:
                return class_ndarray, similarity_ndarray

        # Either an empty attribute set, or no non-zero similarities calculated
        class_ndarray = zeros(0, dtype=object)
        similarity_ndarray = zeros(0)

        return class_ndarray, similarity_ndarray
