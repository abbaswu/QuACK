"""
Typing constraints.
NOTE: This module is intended to be used as a singleton.
"""

import _ast
import ast
import asyncio
import collections
import collections.abc
import datetime
import itertools
import logging
import typing
from collections import Counter
from collections import defaultdict

import switches_singleton
from definitions_to_runtime_terms_mappings_singleton import unwrapped_runtime_functions_to_named_function_definitions
from disjoint_set import DisjointSet
from get_attributes_in_runtime_class import get_attributes_in_runtime_class
from get_dict_for_runtime_class import get_comprehensive_dict_for_runtime_class
from get_unwrapped_constructor import get_unwrapped_constructor
from parameter_lists_and_symbolic_return_values_singleton import nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values
from propagation_task_generation import generate_propagation_tasks, PropagationTask, AttributeAccessPropagationTask, \
    FunctionCallPropagationTask
from relations import NonEquivalenceRelationGraph, NonEquivalenceRelationType
from runtime_term import RuntimeTerm, Instance, UnboundMethod, BoundMethod
from type_ascription import type_ascription
from type_definitions import RuntimeClass, FunctionDefinition, Module, UnwrappedRuntimeFunction, Function
from typeshed_client_ex.client import Client
from typeshed_client_ex.type_definitions import TypeshedClass, \
    get_comprehensive_type_annotations_for_parameters_and_return_values, get_type_annotation_of_self, TypeshedFunction, \
    TypeshedTypeAnnotation, TypeshedTypeVariable, iterate_type_variables_in_type_annotation, to_runtime_class, \
    TypeshedClassDefinition, get_attributes_in_class_definition
from unwrap import unwrap

nodes_to_attribute_counters: defaultdict[_ast.AST, Counter[str]] = defaultdict(Counter)
nodes_to_attribute_counters_lock: asyncio.Lock = asyncio.Lock()

nodes_to_runtime_term_sets: defaultdict[_ast.AST, set[RuntimeTerm]] = defaultdict(set)
nodes_to_runtime_term_sets_lock: asyncio.Lock = asyncio.Lock()

nodes_to_propagation_task_sets: defaultdict[_ast.AST, set[PropagationTask]] = defaultdict(set)
nodes_to_propagation_task_sets_lock: asyncio.Lock = asyncio.Lock()

node_disjoint_set: DisjointSet[_ast.AST] = DisjointSet()
equivalent_set_top_node_non_equivalence_relation_graph: NonEquivalenceRelationGraph = NonEquivalenceRelationGraph()
runtime_term_sharing_node_disjoint_set: DisjointSet[_ast.AST] = DisjointSet()
runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets: defaultdict[
    _ast.AST, set[RuntimeTerm]] = defaultdict(set)
node_disjoint_set_lock: asyncio.Lock = asyncio.Lock()

node_non_equivalence_relation_graph: NonEquivalenceRelationGraph = NonEquivalenceRelationGraph()
node_non_equivalence_relation_graph_lock: asyncio.Lock = asyncio.Lock()

client: Client = Client()


async def create_new_node() -> _ast.AST:
    return _ast.AST()


async def update_attributes(node: _ast.AST, attributes: typing.AbstractSet[str], multiplicity: float = 1.0):
    async with nodes_to_attribute_counters_lock:
        for attribute in attributes:
            nodes_to_attribute_counters[node][attribute] += multiplicity


async def set_node_to_be_instance_of(
        node: _ast.AST,
        runtime_class: RuntimeClass
):
    attributes_in_runtime_class = get_attributes_in_runtime_class(runtime_class)

    # Update attribute counter
    if runtime_class not in (
            type(None),
            type(Ellipsis),
            type(NotImplemented),
    ):
        await update_attributes(node, attributes_in_runtime_class)

    # Update runtime term set
    await add_runtime_terms(node, {Instance(runtime_class)})


async def create_related_node(
        node: _ast.AST,
        relation_type: NonEquivalenceRelationType,
        parameter: typing.Optional[typing.Any] = None
) -> _ast.AST:
    new_node = await create_new_node()
    await add_relation(node, new_node, relation_type, parameter)
    return new_node


async def create_related_node_set(
        node_set: typing.AbstractSet[ast.AST],
        relation_type: NonEquivalenceRelationType,
        parameter: typing.Optional[typing.Any]
) -> frozenset[ast.AST]:
    related_node_list = []
    for node in node_set:
        related_node_list.append(await create_related_node(node, relation_type, parameter))

    related_node_set = frozenset(
        related_node_list
    )

    return related_node_set


async def add_relation(
        from_node: ast.AST,
        to_node: ast.AST,
        relation_type: NonEquivalenceRelationType,
        parameter: typing.Optional[typing.Any] = None
):
    # Add relation in node_non_equivalence_relation_graph
    async with node_non_equivalence_relation_graph_lock:
        node_non_equivalence_relation_graph.add_relation(
            from_node,
            to_node,
            relation_type,
            parameter
        )

    # Induce propagation tasks
    async with nodes_to_runtime_term_sets_lock:
        runtime_term_set = frozenset(nodes_to_runtime_term_sets[from_node])

    if runtime_term_set:
        for propagation_task in generate_propagation_tasks(
                runtime_term_set,
                [(relation_type, parameter)]
        ):
            await add_propagation_task(from_node, propagation_task)

    # Add relation in node_non_equivalence_relation_graph
    # Get induced equivalent set
    async with node_disjoint_set_lock:
        from_node_equivalent_set_top_node = node_disjoint_set.find(from_node)
        to_node_equivalent_set_top_node = node_disjoint_set.find(to_node)

        if not equivalent_set_top_node_non_equivalence_relation_graph.has_relation(
                from_node_equivalent_set_top_node,
                to_node_equivalent_set_top_node,
                relation_type,
                parameter
        ):
            equivalent_set_top_node_non_equivalence_relation_graph.add_relation(
                from_node_equivalent_set_top_node,
                to_node_equivalent_set_top_node,
                relation_type,
                parameter
            )

            induced_equivalent_set = equivalent_set_top_node_non_equivalence_relation_graph.get_out_nodes_with_relation_type_and_parameter(
                from_node_equivalent_set_top_node,
                relation_type,
                parameter
            )
        else:
            induced_equivalent_set = set()

    if (
            relation_type in switches_singleton.valid_relations_to_induce_equivalent_relations
            and induced_equivalent_set
    ):
        await set_equivalent(induced_equivalent_set, propagate_runtime_terms=False)


async def add_argument_of_returned_value_of_relations(
        called_function_node: _ast.AST,
        argument_node_set_list: list[set[_ast.AST]],
        returned_value_node_set: set[_ast.AST]
):
    # Add relation in node_non_equivalence_relation_graph
    async with node_non_equivalence_relation_graph_lock:
        for i, argument_node_set in enumerate(argument_node_set_list):
            for argument_node in argument_node_set:
                node_non_equivalence_relation_graph.add_relation(
                    called_function_node,
                    argument_node,
                    NonEquivalenceRelationType.ArgumentOf,
                    i
                )

        for returned_value_node in returned_value_node_set:
            node_non_equivalence_relation_graph.add_relation(
                called_function_node,
                returned_value_node,
                NonEquivalenceRelationType.ReturnedValueOf
            )

    # Induce propagation tasks
    async with nodes_to_runtime_term_sets_lock:
        runtime_term_set = frozenset(nodes_to_runtime_term_sets[called_function_node])

    if runtime_term_set:
        for propagation_task in generate_propagation_tasks(
                runtime_term_set,
                [
                    *[(NonEquivalenceRelationType.ArgumentOf, i) for i in range(len(argument_node_set_list))],
                    (NonEquivalenceRelationType.ReturnedValueOf, None)
                ]
        ):
            await add_propagation_task(called_function_node, propagation_task)

    # Add relation in node_non_equivalence_relation_graph
    async with node_disjoint_set_lock:
        called_function_node_equivalent_set_top_node = node_disjoint_set.find(called_function_node)

        for i, argument_node_set in enumerate(argument_node_set_list):
            for argument_node in argument_node_set:
                argument_node_equivalent_set_top_node = node_disjoint_set.find(argument_node)
                if not equivalent_set_top_node_non_equivalence_relation_graph.has_relation(
                        called_function_node_equivalent_set_top_node,
                        argument_node_equivalent_set_top_node,
                        NonEquivalenceRelationType.ArgumentOf,
                        i
                ):
                    equivalent_set_top_node_non_equivalence_relation_graph.add_relation(
                        called_function_node_equivalent_set_top_node,
                        argument_node_equivalent_set_top_node,
                        NonEquivalenceRelationType.ArgumentOf,
                        i
                    )

        for returned_value_node in returned_value_node_set:
            returned_value_node_equivalent_set_top_node = node_disjoint_set.find(returned_value_node)
            if not equivalent_set_top_node_non_equivalence_relation_graph.has_relation(
                    called_function_node_equivalent_set_top_node,
                    returned_value_node_equivalent_set_top_node,
                    NonEquivalenceRelationType.ReturnedValueOf
            ):
                equivalent_set_top_node_non_equivalence_relation_graph.add_relation(
                    called_function_node_equivalent_set_top_node,
                    returned_value_node_equivalent_set_top_node,
                    NonEquivalenceRelationType.ReturnedValueOf
                )


async def add_runtime_terms_to_node_only(
        to_node: _ast.AST,
        runtime_term_set: typing.AbstractSet[RuntimeTerm]
):
    propagation_tasks = []

    # Add runtime terms to node
    async with nodes_to_runtime_term_sets_lock, node_non_equivalence_relation_graph_lock:
        runtime_terms_to_add = runtime_term_set - nodes_to_runtime_term_sets[to_node]
        nodes_to_runtime_term_sets[to_node].update(runtime_terms_to_add)

        relation_types_and_parameters = list(
            node_non_equivalence_relation_graph.get_all_relation_types_and_parameters(to_node)
        )

        for propagation_task in generate_propagation_tasks(
                runtime_terms_to_add,
                relation_types_and_parameters
        ):
            propagation_tasks.append(propagation_task)

    for propagation_task in propagation_tasks:
        await add_propagation_task(to_node, propagation_task)


async def add_runtime_terms(
        to_node: ast.AST,
        runtime_term_set: typing.AbstractSet[RuntimeTerm]
):
    # Add runtime terms to other nodes in the runtime term sharing equivalent set
    async with node_disjoint_set_lock:
        runtime_term_sharing_equivalent_set_top_node = runtime_term_sharing_node_disjoint_set.find(to_node)

        runtime_terms_to_add = runtime_term_set - runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets[
            runtime_term_sharing_equivalent_set_top_node]
        runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets[
            runtime_term_sharing_equivalent_set_top_node].update(runtime_terms_to_add)

        all_nodes_in_runtime_term_sharing_equivalent_set = runtime_term_sharing_node_disjoint_set.get_containing_set(
            to_node)

    if runtime_terms_to_add:
        propagation_tasks = []

        for node_in_runtime_term_sharing_equivalent_set in all_nodes_in_runtime_term_sharing_equivalent_set:
            async with nodes_to_runtime_term_sets_lock, node_non_equivalence_relation_graph_lock:
                nodes_to_runtime_term_sets[node_in_runtime_term_sharing_equivalent_set].update(runtime_terms_to_add)

                relation_types_and_parameters = list(
                    node_non_equivalence_relation_graph.get_all_relation_types_and_parameters(
                        node_in_runtime_term_sharing_equivalent_set)
                )

                for propagation_task in generate_propagation_tasks(
                        runtime_terms_to_add,
                        relation_types_and_parameters
                ):
                    propagation_tasks.append((node_in_runtime_term_sharing_equivalent_set, propagation_task))

        for node_in_runtime_term_sharing_equivalent_set, propagation_task in propagation_tasks:
            await add_propagation_task(node_in_runtime_term_sharing_equivalent_set, propagation_task)


async def add_propagation_task(
        node: _ast.AST,
        propagation_task: PropagationTask
):
    add: bool = False

    async with nodes_to_propagation_task_sets_lock:
        if propagation_task not in nodes_to_propagation_task_sets[node]:
            nodes_to_propagation_task_sets[node].add(propagation_task)
            add = True

    if add:
        if isinstance(propagation_task, AttributeAccessPropagationTask):
            if switches_singleton.propagate_attribute_accesses:
                await handle_attribute_access_propagation_task(node, propagation_task)
        elif isinstance(propagation_task, FunctionCallPropagationTask):
            if switches_singleton.propagate_user_defined_function_calls or switches_singleton.propagate_stdlib_function_calls:
                await handle_function_call_propagation_task(node, propagation_task)
        else:
            raise TypeError(f'Unknown propagation task type {type(propagation_task)}!')


async def handle_attribute_access_propagation_task(
        node: _ast.AST,
        attribute_access_propagation_task: AttributeAccessPropagationTask
):
    # Retrieve runtime term and attribute name.
    runtime_term = attribute_access_propagation_task.runtime_term
    attribute_name = attribute_access_propagation_task.attribute_name

    # Get target nodes.
    async with node_non_equivalence_relation_graph_lock:
        target_node_set = node_non_equivalence_relation_graph.get_out_nodes_with_relation_type_and_parameter(
            node,
            NonEquivalenceRelationType.AttrOf,
            attribute_name
        )

    # Get attribute access result.
    attribute_access_result: typing.Optional[RuntimeTerm] = None
    instance_attribute_access_runtime_class: typing.Optional[RuntimeClass] = None

    if isinstance(runtime_term, Module):
        runtime_term_dict = runtime_term.__dict__
        if attribute_name in runtime_term_dict:
            attribute = runtime_term_dict[attribute_name]
            unwrapped_attribute = unwrap(attribute)

            # Module -> Module
            if isinstance(unwrapped_attribute, Module):
                attribute_access_result = unwrapped_attribute
            # Module -> Class
            elif isinstance(unwrapped_attribute, RuntimeClass):
                attribute_access_result = unwrapped_attribute
            # Module -> Function
            elif isinstance(unwrapped_attribute, UnwrappedRuntimeFunction):
                # Function defined within the scope of the project
                if unwrapped_attribute in unwrapped_runtime_functions_to_named_function_definitions:
                    attribute_access_result = unwrapped_runtime_functions_to_named_function_definitions[
                        unwrapped_attribute]
                else:
                    attribute_access_result = unwrapped_attribute
            # Module -> Instance
            elif not callable(unwrapped_attribute):
                instance_attribute_access_runtime_class = type(unwrapped_attribute)
        else:
            logging.error('Cannot statically get attribute `%s` on module %s!', attribute_name, runtime_term)
    elif isinstance(runtime_term, RuntimeClass):
        runtime_term_dict = get_comprehensive_dict_for_runtime_class(runtime_term)
        if attribute_name in runtime_term_dict:
            attribute = runtime_term_dict[attribute_name]
            unwrapped_attribute = unwrap(attribute)

            # Class -> UnboundMethod
            if isinstance(unwrapped_attribute, UnwrappedRuntimeFunction):
                # Function defined within the scope of the project
                if unwrapped_attribute in unwrapped_runtime_functions_to_named_function_definitions:
                    attribute_access_result = UnboundMethod(
                        runtime_term,
                        unwrapped_runtime_functions_to_named_function_definitions[unwrapped_attribute]
                    )
                else:
                    attribute_access_result = UnboundMethod(
                        runtime_term,
                        unwrapped_attribute
                    )
            # Class -> Instance
            elif not callable(unwrapped_attribute):
                instance_attribute_access_runtime_class = type(unwrapped_attribute)
        else:
            logging.error('Cannot statically get attribute `%s` on class %s!', attribute_name, runtime_term)
    elif isinstance(runtime_term, Instance):
        runtime_term_class_dict = get_comprehensive_dict_for_runtime_class(runtime_term.class_)
        if attribute_name in runtime_term_class_dict:
            attribute = runtime_term_class_dict[attribute_name]
            is_staticmethod = isinstance(attribute, staticmethod)
            unwrapped_attribute = unwrap(attribute)

            # Instance -> UnboundMethod
            # Instance -> BoundMethod
            if isinstance(unwrapped_attribute, UnwrappedRuntimeFunction):
                if is_staticmethod:
                    # Function defined within the scope of the project
                    if unwrapped_attribute in unwrapped_runtime_functions_to_named_function_definitions:
                        attribute_access_result = UnboundMethod(
                            runtime_term.class_,
                            unwrapped_runtime_functions_to_named_function_definitions[unwrapped_attribute]
                        )
                    else:
                        attribute_access_result = UnboundMethod(
                            runtime_term.class_,
                            unwrapped_attribute
                        )
                else:
                    # Function defined within the scope of the project
                    if unwrapped_attribute in unwrapped_runtime_functions_to_named_function_definitions:
                        attribute_access_result = BoundMethod(
                            runtime_term,
                            unwrapped_runtime_functions_to_named_function_definitions[unwrapped_attribute]
                        )
                    else:
                        attribute_access_result = BoundMethod(
                            runtime_term,
                            unwrapped_attribute
                        )
        else:
            logging.error('Cannot statically get attribute `%s` on class %s!', attribute_name, runtime_term.class_)

    # Propagate attribute access result.
    if target_node_set and attribute_access_result is not None:
        for target_node in target_node_set:
            await add_runtime_terms(target_node, {attribute_access_result})

    if switches_singleton.propagate_instance_attribute_accesses and instance_attribute_access_runtime_class is not None:
        for target_node in target_node_set:
            await set_node_to_be_instance_of(target_node, instance_attribute_access_runtime_class)


async def get_apparent_argument_node_set_list_and_returned_value_node_set(
        node: _ast.AST
):
    async with node_non_equivalence_relation_graph_lock:
        argument_of_out_edges = node_non_equivalence_relation_graph.get_out_nodes_with_relation_type(
            node,
            NonEquivalenceRelationType.ArgumentOf
        )

        returned_value_of_out_edges = node_non_equivalence_relation_graph.get_out_nodes_with_relation_type_and_parameter(
            node,
            NonEquivalenceRelationType.ReturnedValueOf
        )

    argument_indices: set[int] = {
        parameter
        for parameter in argument_of_out_edges
        if isinstance(parameter, int)
    }

    argument_names: set[str] = {
        parameter
        for parameter in argument_of_out_edges
        if isinstance(parameter, str)
    }

    apparent_argument_node_set_list: list[set[_ast.AST]] = []

    if argument_indices:
        number_of_arguments = max(argument_indices) + 1
        for i in range(number_of_arguments):
            apparent_argument_node_set_list.append(
                argument_of_out_edges.get(i, set())
            )

    argument_name_to_argument_node_set_dict: dict[str, set[_ast.AST]] = dict()
    for argument_name in argument_names:
        argument_name_to_argument_node_set_dict[argument_name] = argument_of_out_edges.get(argument_name, set())

    return apparent_argument_node_set_list, argument_name_to_argument_node_set_dict, returned_value_of_out_edges


async def handle_function_call_propagation_task(
        node: _ast.AST,
        function_call_propagation_task: FunctionCallPropagationTask
):
    runtime_term = function_call_propagation_task.runtime_term

    (
        apparent_argument_node_set_list,
        argument_name_to_argument_node_set_dict,
        returned_value_node_set
    ) = await get_apparent_argument_node_set_list_and_returned_value_node_set(node)

    if isinstance(runtime_term, RuntimeClass):
        # Get the RuntimeClass's constructor
        # apparent_parameter_node_set_list skips the first parameter (self or cls)
        unwrapped_constructor = get_unwrapped_constructor(runtime_term)

        # Function defined within the scope of the project
        if unwrapped_constructor in unwrapped_runtime_functions_to_named_function_definitions:
            named_function_definition = unwrapped_runtime_functions_to_named_function_definitions[unwrapped_constructor]

            (
                parameter_list,
                parameter_name_to_parameter_mapping,
                symbolic_return_value
            ) = nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values[
                named_function_definition
            ]

            if switches_singleton.propagate_user_defined_function_calls:
                await handle_user_defined_function_call(
                    parameter_list[1:],  # Skip the first parameter (self or cls)
                    apparent_argument_node_set_list,
                    parameter_name_to_parameter_mapping,
                    argument_name_to_argument_node_set_dict,
                )
        else:
            # Typeshed has stubs
            try:
                typeshed_name_lookup_result = client.look_up_name(runtime_term.__module__, runtime_term.__name__)
                assert isinstance(typeshed_name_lookup_result, TypeshedClass)

                typeshed_class_definition = client.get_class_definition(typeshed_name_lookup_result)

                if '__init__' in typeshed_class_definition.method_name_to_method_list_dict:
                    (
                        parameter_type_annotation_list,
                        _
                    ) = get_comprehensive_type_annotations_for_parameters_and_return_values(
                        typeshed_class_definition.method_name_to_method_list_dict['__init__']
                    )
                else:
                    (
                        parameter_type_annotation_list,
                        _
                    ) = get_comprehensive_type_annotations_for_parameters_and_return_values(
                        typeshed_class_definition.method_name_to_method_list_dict['__new__']
                    )

                return_value_type_annotation = get_type_annotation_of_self(
                    typeshed_name_lookup_result,
                    typeshed_class_definition.type_variable_list
                )

                if switches_singleton.propagate_stdlib_function_calls:
                    await type_ascription_on_function_call(
                        apparent_argument_node_set_list,
                        returned_value_node_set,
                        # Skip the argument passed to `self` or `cls`
                        parameter_type_annotation_list[1:],
                        return_value_type_annotation,
                    )
            except Exception:
                logging.exception('Cannot lookup constructor for class %s in Typeshed!', runtime_term)

        for returned_value_node in returned_value_node_set:
            await add_runtime_terms(returned_value_node, {Instance(runtime_term)})
    elif isinstance(runtime_term, Function):
        if isinstance(runtime_term, UnwrappedRuntimeFunction):
            # Typeshed has stubs
            try:
                typeshed_name_lookup_result = client.look_up_name(runtime_term.__module__,
                                                                  runtime_term.__name__)
                assert isinstance(typeshed_name_lookup_result, TypeshedFunction)

                typeshed_function_definition_list = client.get_function_definition(
                    typeshed_name_lookup_result)

                (
                    parameter_type_annotation_list,
                    return_value_type_annotation
                ) = get_comprehensive_type_annotations_for_parameters_and_return_values(
                    typeshed_function_definition_list
                )

                # Special handling for isinstance
                if runtime_term == isinstance:
                    # isinstance(object, classinfo)
                    # Only support classinfo being literal classes and literal tuples containing literal classes.
                    runtime_classes: set[RuntimeClass] = set()
                    for classinfo_node in apparent_argument_node_set_list[1]:
                        async with node_non_equivalence_relation_graph_lock, nodes_to_runtime_term_sets_lock:
                            for runtime_term in nodes_to_runtime_term_sets[classinfo_node]:
                                if isinstance(runtime_term, RuntimeClass):
                                    runtime_classes.add(runtime_term)
                            if Instance(tuple) in nodes_to_runtime_term_sets[classinfo_node]:
                                for index, related_node_set in node_non_equivalence_relation_graph.get_out_nodes_with_relation_type(
                                        classinfo_node, NonEquivalenceRelationType.ElementOf
                                ).items():
                                    for related_node in related_node_set:
                                        for related_node_runtime_term in nodes_to_runtime_term_sets[related_node]:
                                            if isinstance(related_node_runtime_term, RuntimeClass):
                                                runtime_classes.add(related_node_runtime_term)

                    for object_node in apparent_argument_node_set_list[0]:
                        for runtime_class in runtime_classes:
                            await update_attributes(
                                object_node,
                                get_attributes_in_runtime_class(runtime_class),
                                1 / len(runtime_classes)
                            )

                # Special handling for delattr, getattr, hasattr, setattr
                elif runtime_term in (delattr, getattr, hasattr, setattr):
                    # delattr(object, name)
                    # getattr(object, name)
                    # getattr(object, name, default)
                    # hasattr(object, name)
                    # setattr(object, name, value)
                    # Only support name being literal strings.
                    attribute_names: set[str] = set()
                    for name_node in apparent_argument_node_set_list[1]:
                        if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
                            attribute_names.add(name_node.value)
                        else:
                            await set_node_to_be_instance_of(name_node, str)

                    for object_node in apparent_argument_node_set_list[0]:
                        await update_attributes(object_node, attribute_names, 1 / len(attribute_names))

                    # Handle getattr
                    if runtime_term == getattr:
                        for object_node in apparent_argument_node_set_list[0]:
                            for returned_value_node in returned_value_node_set:
                                for attribute_name in attribute_names:
                                    await add_relation(
                                        object_node,
                                        returned_value_node,
                                        NonEquivalenceRelationType.AttrOf,
                                        attribute_name
                                    )

                        if len(apparent_argument_node_set_list) >= 3:
                            await set_equivalent(
                                returned_value_node_set | apparent_argument_node_set_list[2]
                            )

                    # Handle hasattr
                    if runtime_term == hasattr:
                        for returned_value_node in returned_value_node_set:
                            await set_node_to_be_instance_of(
                                returned_value_node,
                                bool
                            )

                    # Handle setattr
                    if runtime_term == setattr:
                        for object_node in apparent_argument_node_set_list[0]:
                            for value_node in apparent_argument_node_set_list[2]:
                                for attribute_name in attribute_names:
                                    await add_relation(
                                        object_node,
                                        value_node,
                                        NonEquivalenceRelationType.AttrOf,
                                        attribute_name
                                    )
                else:
                    if switches_singleton.propagate_stdlib_function_calls:
                        await type_ascription_on_function_call(
                            apparent_argument_node_set_list,
                            returned_value_node_set,
                            parameter_type_annotation_list,
                            return_value_type_annotation
                        )
            except Exception:
                logging.exception('Cannot lookup UnwrappedRuntimeFunction %s in Typeshed!', runtime_term)
        elif isinstance(runtime_term, FunctionDefinition):
            (
                parameter_list,
                parameter_name_to_parameter_mapping,
                symbolic_return_value
            ) = nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values[
                runtime_term
            ]

            if switches_singleton.propagate_user_defined_function_calls:
                await handle_user_defined_function_call(
                    parameter_list,
                    apparent_argument_node_set_list,
                    parameter_name_to_parameter_mapping,
                    argument_name_to_argument_node_set_dict,
                    symbolic_return_value,
                    returned_value_node_set,
                )
        else:
            logging.error('Cannot handle function %s!', runtime_term)
    elif isinstance(runtime_term, UnboundMethod):
        if isinstance(runtime_term.function, UnwrappedRuntimeFunction):
            # Typeshed has stubs
            try:
                typeshed_name_lookup_result = client.look_up_name(runtime_term.class_.__module__,
                                                                  runtime_term.class_.__name__)
                assert isinstance(typeshed_name_lookup_result, TypeshedClass)

                typeshed_class_definition = client.get_class_definition(typeshed_name_lookup_result)

                (
                    parameter_type_annotation_list,
                    return_value_type_annotation
                ) = get_comprehensive_type_annotations_for_parameters_and_return_values(
                    typeshed_class_definition.method_name_to_method_list_dict[runtime_term.function.__name__]
                )

                if switches_singleton.propagate_stdlib_function_calls:
                    await type_ascription_on_function_call(
                        apparent_argument_node_set_list,
                        returned_value_node_set,
                        parameter_type_annotation_list,
                        return_value_type_annotation
                    )
            except Exception:
                logging.exception('Cannot lookup UnboundMethod(%s, %s) in Typeshed!', runtime_term.class_,
                                  runtime_term.function)
        elif isinstance(runtime_term.function, FunctionDefinition):
            (
                parameter_list,
                parameter_name_to_parameter_mapping,
                symbolic_return_value
            ) = nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values[
                runtime_term.function
            ]

            if switches_singleton.propagate_user_defined_function_calls:
                await handle_user_defined_function_call(
                    parameter_list,
                    apparent_argument_node_set_list,
                    parameter_name_to_parameter_mapping,
                    argument_name_to_argument_node_set_dict,
                    symbolic_return_value,
                    returned_value_node_set,
                )
        else:
            logging.error('Cannot handle UnboundMethod %s!', runtime_term)
    elif isinstance(runtime_term, Instance):
        runtime_term_class_dict = get_comprehensive_dict_for_runtime_class(runtime_term.class_)
        if '__call__' in runtime_term_class_dict:
            unwrapped_call = unwrap(runtime_term_class_dict['__call__'])
            # Function defined within the scope of the project
            if unwrapped_call in unwrapped_runtime_functions_to_named_function_definitions:
                named_function_definition = unwrapped_runtime_functions_to_named_function_definitions[unwrapped_call]

                (
                    parameter_list,
                    parameter_name_to_parameter_mapping,
                    symbolic_return_value
                ) = nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values[
                    named_function_definition
                ]

                if switches_singleton.propagate_user_defined_function_calls:
                    await handle_user_defined_function_call(
                        parameter_list[1:],  # Skip the first parameter (self)
                        apparent_argument_node_set_list,
                        parameter_name_to_parameter_mapping,
                        argument_name_to_argument_node_set_dict,
                        symbolic_return_value,
                        returned_value_node_set,
                    )
            else:
                # Typeshed has stubs
                try:
                    typeshed_name_lookup_result = client.look_up_name(runtime_term.class_.__module__,
                                                                      runtime_term.class_.__name__)
                    assert isinstance(typeshed_name_lookup_result, TypeshedClass)

                    typeshed_class_definition = client.get_class_definition(typeshed_name_lookup_result)

                    (
                        parameter_type_annotation_list,
                        return_value_type_annotation
                    ) = get_comprehensive_type_annotations_for_parameters_and_return_values(
                        typeshed_class_definition.method_name_to_method_list_dict[unwrapped_call.__name__]
                    )

                    # The type annotation of the `self` parameter
                    parameter_type_annotation_list[0] = get_type_annotation_of_self(
                        typeshed_name_lookup_result,
                        typeshed_class_definition.type_variable_list
                    )

                    # The current node corresponds to the (hidden) argument passed to the `self` parameter
                    argument_node_set_list = [{node}] + apparent_argument_node_set_list

                    if switches_singleton.propagate_stdlib_function_calls:
                        await type_ascription_on_function_call(
                            argument_node_set_list,
                            returned_value_node_set,
                            parameter_type_annotation_list,
                            return_value_type_annotation
                        )
                except Exception:
                    logging.exception('Cannot lookup `__call__` for class %s in Typeshed!', runtime_term.class_)
        else:
            logging.error('Cannot handle call of %s - class does not provide `__call__` method!',
                          runtime_term)
    elif isinstance(runtime_term, BoundMethod):
        # Find node representing the (hidden) argument passed to the `self` parameter
        if isinstance(runtime_term.function, UnwrappedRuntimeFunction):
            method_name = runtime_term.function.__name__
        else:
            method_name = runtime_term.function.name

        # The current node corresponds to the (hidden) argument passed to the `self` parameter
        async with node_non_equivalence_relation_graph_lock:
            self_parameter_argument_node_set = node_non_equivalence_relation_graph.get_in_nodes_with_relation_type_and_parameter(
                node,
                NonEquivalenceRelationType.AttrOf,
                method_name
            )

        if isinstance(runtime_term.function, UnwrappedRuntimeFunction):
            # Typeshed has stubs
            try:
                typeshed_name_lookup_result = client.look_up_name(runtime_term.instance.class_.__module__,
                                                                  runtime_term.instance.class_.__name__)
                assert isinstance(typeshed_name_lookup_result, TypeshedClass)

                typeshed_class_definition = client.get_class_definition(typeshed_name_lookup_result)

                (
                    parameter_type_annotation_list,
                    return_value_type_annotation
                ) = get_comprehensive_type_annotations_for_parameters_and_return_values(
                    typeshed_class_definition.method_name_to_method_list_dict[runtime_term.function.__name__]
                )

                # The type annotation of the `self` parameter
                parameter_type_annotation_list[0] = get_type_annotation_of_self(
                    typeshed_name_lookup_result,
                    typeshed_class_definition.type_variable_list
                )

                argument_node_set_list = [self_parameter_argument_node_set] + apparent_argument_node_set_list

                if switches_singleton.propagate_stdlib_function_calls:
                    await type_ascription_on_function_call(
                        argument_node_set_list,
                        returned_value_node_set,
                        parameter_type_annotation_list,
                        return_value_type_annotation
                    )
            except Exception:
                logging.exception('Cannot lookup %s in Typeshed!', runtime_term)
        elif isinstance(runtime_term.function, FunctionDefinition):
            (
                parameter_list,
                parameter_name_to_parameter_mapping,
                symbolic_return_value
            ) = nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values[
                runtime_term.function
            ]

            if switches_singleton.propagate_user_defined_function_calls:
                await handle_user_defined_function_call(
                    parameter_list[1:],  # Skip the first parameter (self)
                    apparent_argument_node_set_list,
                    parameter_name_to_parameter_mapping,
                    argument_name_to_argument_node_set_dict,
                    symbolic_return_value,
                    returned_value_node_set,
                )
        else:
            logging.error('Cannot handle BoundMethod %s!', runtime_term)


async def type_ascription_on_function_call(
        apparent_argument_node_set_list: typing.Sequence[typing.Set[ast.AST]],
        returned_value_node_set: typing.Set[ast.AST],
        apparent_parameter_type_annotation_list: typing.Sequence[TypeshedTypeAnnotation],
        return_value_type_annotation: TypeshedTypeAnnotation
):
    # Map typeshed type variables to nodes.
    typeshed_type_variables_to_nodes: dict[TypeshedTypeVariable, ast.AST] = dict()

    for parameter_type_annotation in apparent_parameter_type_annotation_list:
        for typeshed_type_variable in iterate_type_variables_in_type_annotation(parameter_type_annotation):
            if typeshed_type_variable not in typeshed_type_variables_to_nodes:
                typeshed_type_variables_to_nodes[typeshed_type_variable] = await create_new_node()

    for typeshed_type_variable in iterate_type_variables_in_type_annotation(return_value_type_annotation):
        if typeshed_type_variable not in typeshed_type_variables_to_nodes:
            typeshed_type_variables_to_nodes[typeshed_type_variable] = await create_new_node()

    # Define callbacks for ascribing typeshed type variables and typeshed classes.
    async def typeshed_type_variable_ascription_callback(
            node_set: frozenset[ast.AST],
            typeshed_type_variable: TypeshedTypeVariable,
            weight: float
    ):
        if not switches_singleton.simplified_type_ascription:
            await set_equivalent(node_set | {typeshed_type_variables_to_nodes[typeshed_type_variable]})

    async def typeshed_class_ascription_callback(
            node_set: frozenset[ast.AST],
            typeshed_class: TypeshedClass,
            weight: float
    ):
        
        runtime_class: RuntimeClass | None = to_runtime_class(typeshed_class)
        if runtime_class is None:
            logging.error('Cannot convert %s to a runtime class!', typeshed_class)

        typeshed_class_definition: TypeshedClassDefinition | None = None
        try:
            typeshed_class_definition = client.get_class_definition(typeshed_class)
        except:
            logging.exception(
                'Cannot retrieve Typeshed class definition for %s!', typeshed_class)

        attribute_set: set[str] = set()

        if typeshed_class_definition is not None:
            attribute_set.update(
                get_attributes_in_class_definition(typeshed_class_definition)
            )

        if runtime_class is not None:
            attribute_set.update(
                get_attributes_in_runtime_class(runtime_class)
            )

        for node in node_set:
            await update_attributes(node, attribute_set, weight)

            if not switches_singleton.simplified_type_ascription and runtime_class is not None:
                await add_runtime_terms(node, {Instance(runtime_class)})

    # Invoke type ascription procedure on apparent arguments and return values.
    for apparent_argument_node_set, apparent_parameter_type_annotation in zip(
            apparent_argument_node_set_list,
            apparent_parameter_type_annotation_list
    ):
        await type_ascription(
            frozenset(apparent_argument_node_set),
            apparent_parameter_type_annotation,
            create_related_node_set,
            typeshed_type_variable_ascription_callback,
            typeshed_class_ascription_callback
        )

    await type_ascription(
        frozenset(returned_value_node_set),
        return_value_type_annotation,
        create_related_node_set,
        typeshed_type_variable_ascription_callback,
        typeshed_class_ascription_callback
    )


async def handle_user_defined_function_call(
        parameter_list: typing.Sequence[ast.arg],
        argument_node_set_list: typing.Sequence[typing.Set[_ast.AST]],
        parameter_name_to_parameter_mapping: typing.Mapping[str, ast.arg],
        argument_name_to_argument_node_set_mapping: typing.Mapping[str, typing.Set[_ast.AST]],
        return_value_node: typing.Optional[_ast.AST] = None,
        returned_value_node_set: typing.Optional[typing.Set[_ast.AST]] = None
):
    # Handle parameter propagation.
    for parameter_node, argument_node_set in zip(parameter_list, argument_node_set_list):
        await handle_matching_parameter_with_argument(parameter_node, argument_node_set)

    # Handle parameter propagation by name.
    parameter_argument_name_matches: set[str] = parameter_name_to_parameter_mapping.keys() & argument_name_to_argument_node_set_mapping.keys()
    for parameter_argument_name_match in parameter_argument_name_matches:
        parameter_node = parameter_name_to_parameter_mapping[parameter_argument_name_match]
        argument_node_set = argument_name_to_argument_node_set_mapping[parameter_argument_name_match]
        await handle_matching_parameter_with_argument(parameter_node, argument_node_set)

    # Handle return value propagation.
    if return_value_node is not None and returned_value_node_set is not None:
        await handle_matching_return_value_with_returned_value(return_value_node, returned_value_node_set)


async def handle_matching_parameter_with_argument(
        parameter_node: ast.arg,
        argument_node_set: set[_ast.AST]
):
    # Contravariance
    # The argument node should be a subtype of the parameter node.
    # i.e., merge the attribute set of the parameter node into that of the argument node.
    await set_equivalent({parameter_node} | argument_node_set, propagate_runtime_terms=False)
    # node_attribute_containment_graph.add_edge(parameter_node, argument_node)


async def handle_matching_return_value_with_returned_value(
        return_value_node: _ast.AST,
        returned_value_node_set: set[_ast.AST]
):
    # Covariance
    # The returned value node should be a subtype of the return value node.
    # i.e., merge the attribute set of the returned value node into that of the return value node.
    await set_equivalent({return_value_node} | returned_value_node_set, propagate_runtime_terms=False)
    # node_attribute_containment_graph.add_edge(returned_value_node, return_value_node)


async def set_equivalent(node_set: typing.AbstractSet[_ast.AST], propagate_runtime_terms: bool = False):
    pairwise_queue: collections.deque[tuple[_ast.AST, _ast.AST]] = collections.deque()
    for first_node, second_node in itertools.pairwise(node_set):
        pairwise_queue.append((first_node, second_node))

    while pairwise_queue:
        first_node, second_node = pairwise_queue.popleft()

        union_happened: bool = False
        target_equivalent_set_top_node: typing.Optional[_ast.AST] = None
        acquirer_equivalent_set_top_node: typing.Optional[_ast.AST] = None

        runtime_term_sharing_node_disjoint_set_union_happened: bool = False
        runtime_term_sharing_target_equivalent_set_top_node: typing.Optional[_ast.AST] = None
        runtime_term_sharing_target_equivalent_set: set[_ast.AST] = set()
        runtime_term_sharing_acquirer_equivalent_set_top_node: typing.Optional[_ast.AST] = None
        runtime_term_sharing_acquirer_equivalent_set: set[_ast.AST] = set()

        induced_equivalent_node_set_list: list[set[_ast.AST]] = []

        runtime_term_sharing_target_equivalent_set_runtime_term_set_delta: set[RuntimeTerm] = set()
        runtime_term_sharing_acquirer_equivalent_set_runtime_term_set_delta: set[RuntimeTerm] = set()

        async with node_disjoint_set_lock:
            # Merge equivalent sets and non-equivalence relations.
            def union_callback(
                    top_element_of_equivalent_set_containing_first_element: _ast.AST,
                    equivalent_set_containing_first_element: set[_ast.AST],
                    top_element_of_equivalent_set_containing_second_element: _ast.AST,
                    equivalent_set_containing_second_element: set[_ast.AST]
            ):
                nonlocal union_happened, target_equivalent_set_top_node, acquirer_equivalent_set_top_node
                union_happened = True
                target_equivalent_set_top_node = top_element_of_equivalent_set_containing_first_element
                acquirer_equivalent_set_top_node = top_element_of_equivalent_set_containing_second_element

            node_disjoint_set.union(
                first_node,
                second_node,
                union_callback
            )

            if union_happened:
                # Non-equivalence relations, induce equivalent node sets
                for (
                        node,
                        relation_type,
                        parameter
                ) in equivalent_set_top_node_non_equivalence_relation_graph.merge_nodes(
                    target_equivalent_set_top_node,
                    acquirer_equivalent_set_top_node
                ):
                    if relation_type in switches_singleton.valid_relations_to_induce_equivalent_relations:
                        induced_equivalent_node_set_list.append(
                            equivalent_set_top_node_non_equivalence_relation_graph.get_out_nodes_with_relation_type_and_parameter(
                                node, relation_type, parameter
                            )
                        )

            # Merge runtime term sharing sets.
            if propagate_runtime_terms:
                def runtime_term_sharing_union_callback(
                        top_element_of_runtime_term_sharing_equivalent_set_containing_first_element: _ast.AST,
                        runtime_term_sharing_equivalent_set_containing_first_element: set[_ast.AST],
                        top_element_of_runtime_term_sharing_equivalent_set_containing_second_element: _ast.AST,
                        runtime_term_sharing_equivalent_set_containing_second_element: set[_ast.AST]
                ):
                    nonlocal runtime_term_sharing_node_disjoint_set_union_happened, runtime_term_sharing_target_equivalent_set_top_node, runtime_term_sharing_target_equivalent_set, runtime_term_sharing_acquirer_equivalent_set_top_node, runtime_term_sharing_acquirer_equivalent_set

                    runtime_term_sharing_node_disjoint_set_union_happened = True
                    runtime_term_sharing_target_equivalent_set_top_node = top_element_of_runtime_term_sharing_equivalent_set_containing_first_element
                    runtime_term_sharing_target_equivalent_set = runtime_term_sharing_equivalent_set_containing_first_element.copy()
                    runtime_term_sharing_acquirer_equivalent_set_top_node = top_element_of_runtime_term_sharing_equivalent_set_containing_second_element
                    runtime_term_sharing_acquirer_equivalent_set = runtime_term_sharing_equivalent_set_containing_second_element.copy()

                runtime_term_sharing_node_disjoint_set.union(
                    first_node,
                    second_node,
                    runtime_term_sharing_union_callback
                )

                if runtime_term_sharing_node_disjoint_set_union_happened:
                    # Runtime term sets
                    runtime_term_sharing_target_equivalent_set_runtime_term_set = \
                        runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets[
                            runtime_term_sharing_target_equivalent_set_top_node
                        ]
                    runtime_term_sharing_acquirer_equivalent_set_runtime_term_set = \
                        runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets[
                            runtime_term_sharing_acquirer_equivalent_set_top_node
                        ]

                    runtime_term_sharing_target_equivalent_set_runtime_term_set_delta = runtime_term_sharing_acquirer_equivalent_set_runtime_term_set - runtime_term_sharing_target_equivalent_set_runtime_term_set
                    runtime_term_sharing_acquirer_equivalent_set_runtime_term_set_delta = runtime_term_sharing_target_equivalent_set_runtime_term_set - runtime_term_sharing_acquirer_equivalent_set_runtime_term_set

                    del runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets[
                        runtime_term_sharing_target_equivalent_set_top_node]
                    runtime_term_sharing_acquirer_equivalent_set_runtime_term_set.update(
                        runtime_term_sharing_acquirer_equivalent_set_runtime_term_set_delta)

        # Propagate induced equivalent node sets
        for induced_equivalence_node_set in induced_equivalent_node_set_list:
            await set_equivalent(induced_equivalence_node_set, propagate_runtime_terms=False)

        # Propagate runtime terms
        if propagate_runtime_terms:
            for runtime_term_sharing_target_equivalent_set_node in runtime_term_sharing_target_equivalent_set:
                await add_runtime_terms_to_node_only(
                    runtime_term_sharing_target_equivalent_set_node,
                    runtime_term_sharing_target_equivalent_set_runtime_term_set_delta
                )

            for runtime_term_sharing_acquirer_equivalent_set_node in runtime_term_sharing_acquirer_equivalent_set:
                await add_runtime_terms_to_node_only(
                    runtime_term_sharing_acquirer_equivalent_set_node,
                    runtime_term_sharing_acquirer_equivalent_set_runtime_term_set_delta
                )
