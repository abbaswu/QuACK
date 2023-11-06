import typing

from relations import NonEquivalenceRelationType, NonEquivalenceRelationTuple
from runtime_term import Instance, UnboundMethod, BoundMethod, RuntimeTerm
from type_definitions import Module, RuntimeClass, Function

AttributeAccessPropagationRuntimeTerm = typing.Union[
    Module,
    RuntimeClass,
    Instance
]


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


PropagationTask = typing.Union[
    AttributeAccessPropagationTask,
    FunctionCallPropagationTask
]


def can_relation_tuple_induce_propagation_task(
        relation_tuple: NonEquivalenceRelationTuple,
        relation_tuple_induces_function_call_propagation_callback: typing.Callable[[], None],
        relation_tuple_induces_attribute_access_propagation_callback: typing.Callable[[str], None]
):
    relation_type, *parameters = relation_tuple
    if relation_type == NonEquivalenceRelationType.AttrOf:
        attribute_name: str = parameters[0]
        relation_tuple_induces_attribute_access_propagation_callback(attribute_name)
    elif relation_type in (NonEquivalenceRelationType.ArgumentOf, NonEquivalenceRelationType.ReturnedValueOf):
        relation_tuple_induces_function_call_propagation_callback()


def generate_propagation_tasks_induced_by_runtime_terms_and_relation_tuples(
        runtime_terms: typing.Iterable[RuntimeTerm],
        relation_tuples: typing.Iterable[NonEquivalenceRelationTuple]
) -> typing.Iterator[PropagationTask]:
    function_call_induced: bool = False
    accessed_attribute_name_set: set[str] = set()

    def relation_tuple_induces_function_call_propagation_callback():
        nonlocal function_call_induced
        function_call_induced = True

    def relation_tuple_induces_attribute_access_propagation_callback(attribute_name: str):
        nonlocal accessed_attribute_name_set
        accessed_attribute_name_set.add(attribute_name)

    for relation_tuple in relation_tuples:
        can_relation_tuple_induce_propagation_task(
            relation_tuple,
            relation_tuple_induces_function_call_propagation_callback,
            relation_tuple_induces_attribute_access_propagation_callback
        )

    for runtime_term in runtime_terms:
        if isinstance(runtime_term, FunctionCallPropagationRuntimeTerm) and function_call_induced:
            yield FunctionCallPropagationTask(runtime_term)
        if isinstance(runtime_term, AttributeAccessPropagationRuntimeTerm) and accessed_attribute_name_set:
            for attribute_name in accessed_attribute_name_set:
                yield AttributeAccessPropagationTask(runtime_term, attribute_name)
