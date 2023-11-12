import logging
import typing

from get_attributes_in_runtime_class import get_attributes_in_runtime_class
from relations import NonEquivalenceRelationType, NonEquivalenceRelationTuple
from runtime_term import Instance, UnboundMethod, BoundMethod, RuntimeTerm
from type_definitions import Module, RuntimeClass, Function


AttributeAccessPropagationRuntimeTerm = typing.Union[
    Module,
    RuntimeClass,
    Instance
]


def get_attributes_in_attribute_access_propagation_runtime_term(
        attribute_access_propagation_runtime_term: AttributeAccessPropagationRuntimeTerm
) -> set[str]:
    if isinstance(attribute_access_propagation_runtime_term, Module):
        return set(attribute_access_propagation_runtime_term.__dict__.keys())
    elif isinstance(attribute_access_propagation_runtime_term, RuntimeClass):
        return get_attributes_in_runtime_class(attribute_access_propagation_runtime_term)
    elif isinstance(attribute_access_propagation_runtime_term, Instance):
        return get_attributes_in_runtime_class(attribute_access_propagation_runtime_term.class_)
    else:
        raise TypeError(f'Unexpected type of attribute_access_propagation_runtime_term: {type(attribute_access_propagation_runtime_term)}')


class AttributeAccessPropagationTask:
    __slots__ = ['runtime_term', 'attribute_name']  # Defining __slots__ to optimize memory usage

    def __init__(self, runtime_term: AttributeAccessPropagationRuntimeTerm, attribute_name: str):
        """
        Initialize the AttributeAccessPropagationTask object.

        :param runtime_term: A AttributeAccessPropagationRuntimeTerm object.
        :param attribute_name: A string representing the attribute name.
        """
        self.runtime_term = runtime_term  # Should be an instance of RuntimeTerm
        self.attribute_name = attribute_name  # Should be a string

    def __hash__(self) -> int:
        """
        Return a hash value for the object.

        The hash value is computed based on the runtime_term and attribute_name to make the object hashable.
        """
        return hash((self.runtime_term, self.attribute_name))

    def __eq__(self, other: object) -> bool:
        """
        Determine equality with another object.

        Two AttributeAccessPropagationTask objects are considered equal if their runtime_term and attribute_name are equal.

        :param other: Another object to compare with.
        :return: True if objects are equal, otherwise False.
        """
        if not isinstance(other, AttributeAccessPropagationTask):
            return False
        return (self.runtime_term, self.attribute_name) == (other.runtime_term, other.attribute_name)

    def __repr__(self):
        return f'AttributeAccessPropagationTask({self.runtime_term}, {self.attribute_name})'


FunctionCallPropagationRuntimeTerm = typing.Union[
    RuntimeClass,
    Function,
    UnboundMethod,
    Instance,
    BoundMethod
]


class FunctionCallPropagationTask:
    __slots__ = ['runtime_term']  # Defining __slots__ to optimize memory usage

    def __init__(self, runtime_term: FunctionCallPropagationRuntimeTerm):
        """
        Initialize the FunctionCallPropagationTask object.

        :param runtime_term: A FunctionCallPropagationRuntimeTerm object.
        """
        self.runtime_term = runtime_term  # Should be an instance of RuntimeTerm

    def __hash__(self) -> int:
        """
        Return a hash value for the object.

        The hash value is computed based on the runtime_term to make the object hashable.
        """
        return hash(self.runtime_term)

    def __eq__(self, other: object) -> bool:
        """
        Determine equality with another object.

        Two FunctionCallPropagationTask objects are considered equal if their runtime_term is equal.

        :param other: Another object to compare with.
        :return: True if objects are equal, otherwise False.
        """
        if not isinstance(other, FunctionCallPropagationTask):
            return False
        return self.runtime_term == other.runtime_term

    def __repr__(self):
        return f'FunctionCallPropagationTask({self.runtime_term})'


PropagationTask = typing.Union[
    AttributeAccessPropagationTask,
    FunctionCallPropagationTask
]


def generate_propagation_tasks(
        runtime_terms: typing.Iterable[RuntimeTerm],
        relation_types_and_parameters: typing.Iterable[tuple[NonEquivalenceRelationType, typing.Optional[object]]]
) -> typing.Iterator[PropagationTask]:
    function_call_induced: bool = False
    accessed_attribute_name_set: set[str] = set()

    for relation_type, parameter in relation_types_and_parameters:
        if relation_type == NonEquivalenceRelationType.AttrOf:
            assert isinstance(parameter, str)
            accessed_attribute_name_set.add(parameter)
        elif relation_type in (NonEquivalenceRelationType.ArgumentOf, NonEquivalenceRelationType.ReturnedValueOf):
            function_call_induced = True

    for runtime_term in runtime_terms:
        if isinstance(runtime_term, FunctionCallPropagationRuntimeTerm) and function_call_induced:
            logging.info('Function call propagation induced by %s', runtime_term)
            yield FunctionCallPropagationTask(runtime_term)
        if isinstance(runtime_term, AttributeAccessPropagationRuntimeTerm) and accessed_attribute_name_set:
            for attribute_name in accessed_attribute_name_set & get_attributes_in_attribute_access_propagation_runtime_term(
                    runtime_term
            ):
                logging.info('%s-attribute access propagation induced by %s', attribute_name, runtime_term)
                yield AttributeAccessPropagationTask(runtime_term, attribute_name)
