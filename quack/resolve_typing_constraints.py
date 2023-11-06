import ast
import itertools
import logging
import typing
from collections import defaultdict, deque

import networkx as nx

from attribute_counter import AttributeCounter
from disjoint_set import DisjointSet
from get_attributes_in_runtime_class import get_attributes_in_runtime_class
from get_dict_for_runtime_class import get_comprehensive_dict_for_runtime_class
from get_unwrapped_constructor import get_unwrapped_constructor
from graph_condensation import graph_condensation
from propagation_task_generation import PropagationTask, \
    generate_propagation_tasks_induced_by_runtime_terms_and_relation_tuples, AttributeAccessPropagationTask, \
    FunctionCallPropagationTask
from relations import NonEquivalenceRelationType, NonEquivalenceRelationTuple, EquivalenceRelationGraph, \
    NonEquivalenceRelationGraph
from runtime_term import RuntimeTerm, UnboundMethod, Instance, BoundMethod
from scoped_node_visitor import NodeProvidingScope
from topological_sort_edges import topological_sort_edges
from type_ascription import type_ascription
from type_definitions import Module, RuntimeClass, UnwrappedRuntimeFunction, NamedFunctionDefinition, Function, \
    FunctionDefinition
from typeshed_client_ex.client import Client
from typeshed_client_ex.type_definitions import TypeshedClass, get_type_annotation_of_self, \
    get_comprehensive_type_annotations_for_parameters_and_return_values, TypeshedTypeAnnotation, TypeshedFunction, \
    TypeshedTypeVariable, iterate_type_variables_in_type_annotation, to_runtime_class, TypeshedClassDefinition, \
    get_attributes_in_class_definition
from unwrap import unwrap


def resolve_typing_constraints(
        unwrapped_runtime_functions_to_named_function_definitions: typing.Mapping[
            UnwrappedRuntimeFunction,
            NamedFunctionDefinition
        ],
        nodes_to_attribute_counters: defaultdict[ast.AST, AttributeCounter],
        nodes_to_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]],
        nodes_providing_scope_to_parameter_lists_and_return_value_sets: defaultdict[
            NodeProvidingScope,
            tuple[list[ast.arg], set[ast.AST]]
        ],
        equivalence_relation_graph: EquivalenceRelationGraph,
        non_equivalence_relation_graph: NonEquivalenceRelationGraph,
        typeshed_client: Client
):
    """
    Resolve all *induced* equivalence relations
    Merge attribute counters and runtime term sets of equivalent nodes
    And propagte runtime terms
    After collecting preliminary typing constraints based on the semantics of each AST node.
    """
    equivalent_node_disjoint_set: DisjointSet[ast.AST] = DisjointSet()
    new_equivalence_relation_graph: EquivalenceRelationGraph = equivalence_relation_graph.copy()
    new_nodes_to_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]] = nodes_to_runtime_term_sets.copy()
    new_nodes_to_attribute_counters: defaultdict[ast.AST, AttributeCounter] = nodes_to_attribute_counters.copy()
    new_non_equivalence_relation_graph: NonEquivalenceRelationGraph = non_equivalence_relation_graph.copy()

    def add_new_dummy_node() -> ast.AST:
        dummy_node = ast.AST()

        equivalent_node_disjoint_set.find(dummy_node)
        new_equivalence_relation_graph.add_node(dummy_node)
        new_nodes_to_runtime_term_sets[dummy_node] = set()
        new_nodes_to_attribute_counters[dummy_node] = AttributeCounter()
        new_non_equivalence_relation_graph.add_node(dummy_node)

        return dummy_node

    # Fixpoint Iteration to resolve all *induced* Equivalence Relations.

    nodes_to_processed_propagation_task_sets: defaultdict[ast.AST, set[PropagationTask]] = defaultdict(set)

    equivalent_node_pair_queue: deque[tuple[ast.AST, ast.AST]] = deque()
    node_propagation_task_queue: deque[tuple[ast.AST, PropagationTask]] = deque()

    equivalent_node_pair_queue.extend(equivalence_relation_graph.edges.keys())

    for node, relation_tuple_to_related_nodes in new_non_equivalence_relation_graph.iterate_nodes_and_out_edges_by_relation_tuple():
        for propagation_task in generate_propagation_tasks_induced_by_runtime_terms_and_relation_tuples(
                new_nodes_to_runtime_term_sets[node],
                relation_tuple_to_related_nodes.keys()
        ):
            node_propagation_task_queue.append((node, propagation_task))

    # Used to store the covariance and contravariance relations between type variables.
    # This translates into one node's attribute counter containing another node's.
    # The edge A -> B in the graph signifies that B's attribute counter contains A's.
    node_attribute_containment_graph: nx.DiGraph = nx.DiGraph()

    def get_or_create_related_nodes(
            from_set: typing.AbstractSet[ast.AST],
            relation_tuple: NonEquivalenceRelationTuple
    ) -> frozenset[ast.AST]:
        out_edges: set[ast.AST] = set()

        for from_ in from_set:
            out_edges_by_relation_tuple = new_non_equivalence_relation_graph.get_out_edges_by_relation_tuple(from_)
            out_edges.update(out_edges_by_relation_tuple[relation_tuple])

        if out_edges:
            return frozenset(out_edges)
        else:
            to = add_new_dummy_node()

            for from_ in from_set:
                # Induce propagation tasks
                for propagation_task in generate_propagation_tasks_induced_by_runtime_terms_and_relation_tuples(
                        new_nodes_to_runtime_term_sets[from_],
                        [relation_tuple]
                ):
                    node_propagation_task_queue.append((from_, propagation_task))

                # Add relation in graph
                new_non_equivalence_relation_graph.add_relation(
                    from_,
                    to,
                    *relation_tuple
                )

            return frozenset({to})

    def add_relation(
            from_: ast.AST,
            to: ast.AST,
            relation_tuple: NonEquivalenceRelationTuple
    ):
        # Induce propagation tasks
        for propagation_task in generate_propagation_tasks_induced_by_runtime_terms_and_relation_tuples(
                new_nodes_to_runtime_term_sets[from_],
                [relation_tuple]
        ):
            node_propagation_task_queue.append((from_, propagation_task))

        # Add relation in graph
        new_non_equivalence_relation_graph.add_relation(
            from_,
            to,
            *relation_tuple
        )

        return to

    def add_runtime_terms(
            to: ast.AST,
            runtime_terms: typing.Collection[RuntimeTerm]
    ):
        # Induce propagation tasks
        for propagation_task in generate_propagation_tasks_induced_by_runtime_terms_and_relation_tuples(
                runtime_terms,
                new_non_equivalence_relation_graph.get_out_edges_by_relation_tuple(to)
        ):
            node_propagation_task_queue.append((to, propagation_task))

        # Add runtime term
        new_nodes_to_runtime_term_sets[to].update(runtime_terms)

    def handle_attribute_access_propagation_task(
            node: ast.AST,
            attribute_access_propagation_task: AttributeAccessPropagationTask
    ):
        # Retrieve runtime term and attribute name.
        runtime_term = attribute_access_propagation_task.runtime_term
        attribute_name = attribute_access_propagation_task.attribute_name

        def propagate_attribute_access_result(
                attribute_access_result: RuntimeTerm
        ):
            nonlocal node, attribute_name

            for target_node in new_non_equivalence_relation_graph.get_out_edges_by_relation_tuple(node)[
                (NonEquivalenceRelationType.AttrOf, attribute_name)
            ]:
                add_runtime_terms(target_node, [attribute_access_result])

        if isinstance(runtime_term, Module):
            runtime_term_dict = runtime_term.__dict__
            if attribute_name in runtime_term_dict:
                attribute = runtime_term_dict[attribute_name]
                unwrapped_attribute = unwrap(attribute)

                # Module -> Module
                # Module -> Class
                if isinstance(unwrapped_attribute, (Module, RuntimeClass)):
                    propagate_attribute_access_result(unwrapped_attribute)
                # Module -> Function
                elif isinstance(unwrapped_attribute, UnwrappedRuntimeFunction):
                    # Function defined within the scope of the project
                    if unwrapped_attribute in unwrapped_runtime_functions_to_named_function_definitions:
                        propagate_attribute_access_result(
                            unwrapped_runtime_functions_to_named_function_definitions[unwrapped_attribute])
                    else:
                        propagate_attribute_access_result(unwrapped_attribute)
                else:
                    logging.error(
                        'Cannot propagate accessing attribute `%s` on module %s (%s) - it is not a module or a class!',
                        attribute_name,
                        runtime_term, unwrapped_attribute)
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
                        propagate_attribute_access_result(
                            UnboundMethod(
                                runtime_term,
                                unwrapped_runtime_functions_to_named_function_definitions[unwrapped_attribute]
                            )
                        )
                    else:
                        propagate_attribute_access_result(
                            UnboundMethod(
                                runtime_term,
                                unwrapped_attribute
                            )
                        )
                else:
                    logging.error('Cannot propagate accessing attribute `%s` on class %s (%s) - it is not a method!',
                                  attribute_name,
                                  runtime_term, unwrapped_attribute)
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
                            propagate_attribute_access_result(
                                UnboundMethod(
                                    runtime_term.class_,
                                    unwrapped_runtime_functions_to_named_function_definitions[unwrapped_attribute]
                                )
                            )
                        else:
                            propagate_attribute_access_result(
                                UnboundMethod(
                                    runtime_term.class_,
                                    unwrapped_attribute
                                )
                            )
                    else:
                        # Function defined within the scope of the project
                        if unwrapped_attribute in unwrapped_runtime_functions_to_named_function_definitions:
                            propagate_attribute_access_result(
                                BoundMethod(
                                    runtime_term,
                                    unwrapped_runtime_functions_to_named_function_definitions[unwrapped_attribute]
                                )
                            )
                        else:
                            propagate_attribute_access_result(
                                BoundMethod(
                                    runtime_term,
                                    unwrapped_attribute
                                )
                            )
                else:
                    logging.error('Cannot propagate accessing attribute `%s` on class %s (%s) - it is not a method!',
                                  attribute_name,
                                  runtime_term.class_, unwrapped_attribute)
            else:
                logging.error('Cannot statically get attribute `%s` on class %s!', attribute_name, runtime_term.class_)

    def update_attribute_counter_with_attribute_set(
            node: ast.AST,
            attribute_set: set[str],
            multiplicity: float = 1.0
    ):
        for attribute in attribute_set:
            new_nodes_to_attribute_counters[node][attribute] += multiplicity

    def type_ascription_on_function_call(
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
                    typeshed_type_variables_to_nodes[typeshed_type_variable] = add_new_dummy_node()

        for typeshed_type_variable in iterate_type_variables_in_type_annotation(return_value_type_annotation):
            if typeshed_type_variable not in typeshed_type_variables_to_nodes:
                typeshed_type_variables_to_nodes[typeshed_type_variable] = add_new_dummy_node()

        # Define callbacks for ascribing typeshed type variables and typeshed classes.
        def typeshed_type_variable_ascription_callback(
                node_set: frozenset[ast.AST],
                typeshed_type_variable: TypeshedTypeVariable,
                weight: float
        ):
            for node in node_set:
                equivalent_node_pair_queue.append(
                    (node, typeshed_type_variables_to_nodes[typeshed_type_variable]))

        def typeshed_class_ascription_callback(
                node_set: frozenset[ast.AST],
                typeshed_class: TypeshedClass,
                weight: float
        ):
            runtime_class: RuntimeClass | None = to_runtime_class(typeshed_class)
            if runtime_class is None:
                logging.error('Cannot convert %s to a runtime class!', typeshed_class)

            typeshed_class_definition: TypeshedClassDefinition | None = None
            try:
                typeshed_class_definition = typeshed_client.get_class_definition(typeshed_class)
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
                update_attribute_counter_with_attribute_set(
                    node,
                    attribute_set,
                    weight
                )

                if runtime_class is not None:
                    add_runtime_terms(node, [Instance(runtime_class)])

        # Invoke type ascription procedure on apparent arguments and return values.
        for apparent_argument_node_set, apparent_parameter_type_annotation in zip(
                apparent_argument_node_set_list,
                apparent_parameter_type_annotation_list
        ):
            type_ascription(
                frozenset(apparent_argument_node_set),
                apparent_parameter_type_annotation,
                get_or_create_related_nodes,
                typeshed_type_variable_ascription_callback,
                typeshed_class_ascription_callback,
                True
            )

        type_ascription(
            frozenset(returned_value_node_set),
            return_value_type_annotation,
            get_or_create_related_nodes,
            typeshed_type_variable_ascription_callback,
            typeshed_class_ascription_callback,
            True
        )

    def get_apparent_argument_node_set_list(
            node: ast.AST,
            number_of_apparent_argument_nodes: int
    ) -> list[set[ast.AST]]:
        apparent_argument_node_set_list = []

        out_edges_by_relation_tuple = new_non_equivalence_relation_graph.get_out_edges_by_relation_tuple(node)

        for i in range(number_of_apparent_argument_nodes):
            argument_node_set: set[ast.AST] = set()

            relation_tuple = (NonEquivalenceRelationType.ArgumentOf, i)
            if relation_tuple in out_edges_by_relation_tuple:
                argument_node_set.update(out_edges_by_relation_tuple[relation_tuple])

            apparent_argument_node_set_list.append(argument_node_set)

        return apparent_argument_node_set_list

    def get_returned_value_node_set(
            node: ast.AST
    ) -> set[ast.AST]:
        returned_value_node_set = set()

        out_edges_by_relation_tuple = new_non_equivalence_relation_graph.get_out_edges_by_relation_tuple(node)

        relation_tuple = (NonEquivalenceRelationType.ReturnedValueOf,)
        if relation_tuple in out_edges_by_relation_tuple:
            returned_value_node_set.update(out_edges_by_relation_tuple[relation_tuple])

        return returned_value_node_set

    def handle_matching_parameter_with_argument(
            parameter_node: ast.AST,
            argument_node: ast.AST
    ):
        # Contravariance
        # The argument node should be a subtype of the parameter node.
        # i.e., merge the attribute set of the parameter node into that of the argument node.
        node_attribute_containment_graph.add_edge(parameter_node, argument_node)

    def handle_matching_return_value_with_returned_value(
            return_value_node: ast.AST,
            returned_value_node: ast.AST
    ):
        # Covariance
        # The returned value node should be a subtype of the return value node.
        # i.e., merge the attribute set of the returned value node into that of the return value node.
        node_attribute_containment_graph.add_edge(returned_value_node, return_value_node)

    def handle_function_call_propagation_task(
            node: ast.AST,
            function_call_propagation_task: FunctionCallPropagationTask
    ):
        runtime_term = function_call_propagation_task.runtime_term

        returned_value_node_set = get_returned_value_node_set(node)

        if isinstance(runtime_term, RuntimeClass):
            # Get the RuntimeClass's constructor
            # apparent_parameter_node_set_list skips the first parameter (self or cls)
            unwrapped_constructor = get_unwrapped_constructor(runtime_term)

            # Function defined within the scope of the project
            if unwrapped_constructor in unwrapped_runtime_functions_to_named_function_definitions:
                (
                    parameter_list,
                    _
                ) = nodes_providing_scope_to_parameter_lists_and_return_value_sets[
                    unwrapped_runtime_functions_to_named_function_definitions[unwrapped_constructor]
                ]

                # Skip the argument passed to `self` or `cls`
                apparent_argument_node_set_list = get_apparent_argument_node_set_list(node, len(parameter_list) - 1)

                for parameter_node, argument_node_set in zip(parameter_list[1:], apparent_argument_node_set_list):
                    for argument_node in argument_node_set:
                        handle_matching_parameter_with_argument(parameter_node, argument_node)
            else:
                # Typeshed has stubs
                try:
                    typeshed_name_lookup_result = typeshed_client.look_up_name(runtime_term.__module__,
                                                                               runtime_term.__name__)
                    assert isinstance(typeshed_name_lookup_result, TypeshedClass)

                    typeshed_class_definition = typeshed_client.get_class_definition(typeshed_name_lookup_result)

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

                    # Skip the argument passed to `self` or `cls`
                    apparent_argument_node_set_list = get_apparent_argument_node_set_list(node,
                                                                                          len(parameter_type_annotation_list) - 1)

                    type_ascription_on_function_call(
                        apparent_argument_node_set_list,
                        returned_value_node_set,
                        parameter_type_annotation_list[1:],
                        return_value_type_annotation
                    )
                except Exception:
                    logging.exception('Cannot lookup constructor for class %s in Typeshed!', runtime_term)

            for returned_value_node in returned_value_node_set:
                add_runtime_terms(returned_value_node, [Instance(runtime_term)])
        elif isinstance(runtime_term, Function):
            if isinstance(runtime_term, UnwrappedRuntimeFunction):
                # Typeshed has stubs
                try:
                    typeshed_name_lookup_result = typeshed_client.look_up_name(runtime_term.__module__,
                                                                               runtime_term.__name__)
                    assert isinstance(typeshed_name_lookup_result, TypeshedFunction)

                    typeshed_function_definition_list = typeshed_client.get_function_definition(
                        typeshed_name_lookup_result)

                    (
                        parameter_type_annotation_list,
                        return_value_type_annotation
                    ) = get_comprehensive_type_annotations_for_parameters_and_return_values(
                        typeshed_function_definition_list
                    )

                    apparent_argument_node_set_list = get_apparent_argument_node_set_list(node,
                                                                                          len(parameter_type_annotation_list))

                    # Special handling for isinstance
                    if runtime_term == isinstance:
                        # isinstance(object, classinfo)
                        # Only support classinfo being literal classes and literal tuples containing literal classes.
                        runtime_classes: set[RuntimeClass] = set()
                        for classinfo_node in apparent_argument_node_set_list[1]:
                            for runtime_term in new_nodes_to_runtime_term_sets[classinfo_node]:
                                if isinstance(runtime_term, RuntimeClass):
                                    runtime_classes.add(runtime_term)
                            if Instance(tuple) in new_nodes_to_runtime_term_sets[classinfo_node]:
                                for relation_tuple, related_node_set in new_non_equivalence_relation_graph.get_out_edges_by_relation_tuple(
                                        classinfo_node).items():
                                    relation_type, *_ = relation_tuple
                                    if relation_type == NonEquivalenceRelationType.ElementOf:
                                        for related_node in related_node_set:
                                            for related_node_runtime_term in new_nodes_to_runtime_term_sets[
                                                related_node]:
                                                if isinstance(related_node_runtime_term, RuntimeClass):
                                                    runtime_classes.add(related_node_runtime_term)

                        for object_node in apparent_argument_node_set_list[0]:
                            for runtime_class in runtime_classes:

                                update_attribute_counter_with_attribute_set(
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
                            if isinstance(name_node, ast.Constant):
                                if isinstance(name_node.value, str):
                                    attribute_names.add(name_node.value)

                        for object_node in apparent_argument_node_set_list[0]:
                            for attribute_name in attribute_names:
                                new_nodes_to_attribute_counters[object_node][attribute_name] += 1 / len(
                                    attribute_names)

                        # Handle getattr
                        if runtime_term == getattr:
                            for object_node in apparent_argument_node_set_list[0]:
                                for returned_value_node in returned_value_node_set:
                                    for attribute_name in attribute_names:
                                        add_relation(
                                            object_node,
                                            returned_value_node,
                                            (NonEquivalenceRelationType.AttrOf, attribute_name)
                                        )

                            if len(apparent_argument_node_set_list) >= 3:
                                for default_node in apparent_argument_node_set_list[2]:
                                    for returned_value_node in returned_value_node_set:
                                        equivalent_node_pair_queue.append((default_node, returned_value_node))

                        # Handle hasattr
                        if runtime_term == hasattr:
                            for returned_value_node in returned_value_node_set:
                                update_attribute_counter_with_attribute_set(returned_value_node,
                                                                            get_attributes_in_runtime_class(bool))

                        # Handle setattr
                        if runtime_term == setattr:
                            for object_node in apparent_argument_node_set_list[0]:
                                for value_node in apparent_argument_node_set_list[2]:
                                    for attribute_name in attribute_names:
                                        add_relation(
                                            object_node,
                                            value_node,
                                            (NonEquivalenceRelationType.AttrOf, attribute_name)
                                        )
                    else:
                        type_ascription_on_function_call(
                            get_apparent_argument_node_set_list(node, len(parameter_type_annotation_list)),
                            returned_value_node_set,
                            parameter_type_annotation_list,
                            return_value_type_annotation
                        )
                except Exception:
                    logging.exception('Cannot lookup UnwrappedRuntimeFunction %s in Typeshed!', runtime_term)
            elif isinstance(runtime_term, FunctionDefinition):
                (
                    parameter_list,
                    return_value_set
                ) = nodes_providing_scope_to_parameter_lists_and_return_value_sets[
                    runtime_term
                ]

                apparent_argument_node_set_list = get_apparent_argument_node_set_list(node, len(parameter_list))

                for parameter_node, argument_node_set in zip(parameter_list, apparent_argument_node_set_list):
                    for argument_node in argument_node_set:
                        handle_matching_parameter_with_argument(parameter_node, argument_node)
                for return_value_node in return_value_set:
                    for returned_value_node in returned_value_node_set:
                        handle_matching_return_value_with_returned_value(return_value_node, returned_value_node)
            else:
                logging.error('Cannot handle function %s!', runtime_term)
        elif isinstance(runtime_term, UnboundMethod):
            if isinstance(runtime_term.function, UnwrappedRuntimeFunction):
                # Typeshed has stubs
                try:
                    typeshed_name_lookup_result = typeshed_client.look_up_name(runtime_term.class_.__module__,
                                                                               runtime_term.class_.__name__)
                    assert isinstance(typeshed_name_lookup_result, TypeshedClass)

                    typeshed_class_definition = typeshed_client.get_class_definition(typeshed_name_lookup_result)

                    (
                        parameter_type_annotation_list,
                        return_value_type_annotation
                    ) = get_comprehensive_type_annotations_for_parameters_and_return_values(
                        typeshed_class_definition.method_name_to_method_list_dict[runtime_term.function.__name__]
                    )

                    apparent_argument_node_set_list = get_apparent_argument_node_set_list(node,
                                                                                          len(parameter_type_annotation_list))

                    type_ascription_on_function_call(
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
                    return_value_set
                ) = nodes_providing_scope_to_parameter_lists_and_return_value_sets[
                    runtime_term.function
                ]

                apparent_argument_node_set_list = get_apparent_argument_node_set_list(node, len(parameter_list))

                for parameter_node, argument_node_set in zip(parameter_list, apparent_argument_node_set_list):
                    for argument_node in argument_node_set:
                        handle_matching_parameter_with_argument(parameter_node, argument_node)

                for return_value_node in return_value_set:
                    for returned_value_node in returned_value_node_set:
                        handle_matching_return_value_with_returned_value(return_value_node, returned_value_node)
            else:
                logging.error('Cannot handle UnboundMethod %s!', runtime_term)
        elif isinstance(runtime_term, Instance):
            runtime_term_class_dict = get_comprehensive_dict_for_runtime_class(runtime_term.class_)
            if '__call__' in runtime_term_class_dict:
                unwrapped_call = unwrap(runtime_term_class_dict['__call__'])
                # Function defined within the scope of the project
                if unwrapped_call in unwrapped_runtime_functions_to_named_function_definitions:
                    (
                        parameter_list,
                        return_value_set
                    ) = nodes_providing_scope_to_parameter_lists_and_return_value_sets[
                        unwrapped_runtime_functions_to_named_function_definitions[unwrapped_call]
                    ]

                    # The current node corresponds to the (hidden) argument passed to the `self` parameter
                    if parameter_list:
                        equivalent_node_pair_queue.append((parameter_list[0], node))

                        apparent_argument_node_set_list = get_apparent_argument_node_set_list(node,
                                                                                              len(parameter_list) - 1)

                        for parameter_node, argument_node_set in zip(parameter_list[1:],
                                                                     apparent_argument_node_set_list):
                            for argument_node in argument_node_set:
                                handle_matching_parameter_with_argument(parameter_node, argument_node)

                    for return_value_node in return_value_set:
                        for returned_value_node in returned_value_node_set:
                            handle_matching_return_value_with_returned_value(return_value_node, returned_value_node)
                else:
                    # Typeshed has stubs
                    try:
                        typeshed_name_lookup_result = typeshed_client.look_up_name(runtime_term.class_.__module__,
                                                                                   runtime_term.class_.__name__)
                        assert isinstance(typeshed_name_lookup_result, TypeshedClass)

                        typeshed_class_definition = typeshed_client.get_class_definition(typeshed_name_lookup_result)

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
                        argument_node_set_list = [{node}]
                        argument_node_set_list.extend(
                            get_apparent_argument_node_set_list(node, len(parameter_type_annotation_list) - 1)
                        )

                        type_ascription_on_function_call(
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
            in_edges_by_relation_tuple = new_non_equivalence_relation_graph.get_in_edges_by_relation_tuple(node)
            if (NonEquivalenceRelationType.AttrOf, method_name) in in_edges_by_relation_tuple:
                self_parameter_argument_node_set = in_edges_by_relation_tuple[
                    (NonEquivalenceRelationType.AttrOf, method_name)]
            else:
                self_parameter_argument_node_set = set()

            if isinstance(runtime_term.function, UnwrappedRuntimeFunction):
                # Typeshed has stubs
                try:
                    typeshed_name_lookup_result = typeshed_client.look_up_name(runtime_term.instance.class_.__module__,
                                                                               runtime_term.instance.class_.__name__)
                    assert isinstance(typeshed_name_lookup_result, TypeshedClass)

                    typeshed_class_definition = typeshed_client.get_class_definition(typeshed_name_lookup_result)

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

                    argument_node_set_list = [self_parameter_argument_node_set]
                    argument_node_set_list.extend(
                        get_apparent_argument_node_set_list(node, len(parameter_type_annotation_list) - 1)
                    )

                    type_ascription_on_function_call(
                        argument_node_set_list,
                        returned_value_node_set,
                        parameter_type_annotation_list,
                        return_value_type_annotation
                    )
                except Exception:
                    logging.exception('Cannot lookup %s in Typeshed!',
                                      runtime_term)
            elif isinstance(runtime_term.function, FunctionDefinition):
                (
                    parameter_list,
                    return_value_set
                ) = nodes_providing_scope_to_parameter_lists_and_return_value_sets[
                    runtime_term.function
                ]

                if parameter_list:
                    for self_parameter_argument_node in self_parameter_argument_node_set:
                        equivalent_node_pair_queue.append((parameter_list[0], self_parameter_argument_node))

                    apparent_argument_node_set_list = get_apparent_argument_node_set_list(node, len(parameter_list) - 1)

                    for parameter_node, argument_node_set in zip(parameter_list[1:], apparent_argument_node_set_list):
                        for argument_node in argument_node_set:
                            handle_matching_parameter_with_argument(parameter_node, argument_node)

                for return_value_node in return_value_set:
                    for returned_value_node in returned_value_node_set:
                        handle_matching_return_value_with_returned_value(return_value_node, returned_value_node)

            else:
                logging.error('Cannot handle BoundMethod %s!', runtime_term)

    def propagate_runtime_terms(
            node_set: set[ast.AST]
    ):
        all_runtime_terms: set[RuntimeTerm] = set()

        for node in node_set:
            all_runtime_terms.update(new_nodes_to_runtime_term_sets[node])

        for node in node_set:
            add_runtime_terms(node, all_runtime_terms - new_nodes_to_runtime_term_sets[node])

    def add_induced_equivalent_relations(
            node_set: set[ast.AST]
    ):
        merged_out_edges_by_relation_tuple: defaultdict[NonEquivalenceRelationTuple, set[ast.AST]] = defaultdict(set)

        for node in node_set:
            out_edges_by_relation_tuple = new_non_equivalence_relation_graph.get_out_edges_by_relation_tuple(node)

            # Do not consider ArgumentOf relations.
            # Counterexample: isinstance(x, int), isinstance(y, str), we cannot say that x and y are equivalent.
            for relation_tuple, out_edges in out_edges_by_relation_tuple.items():
                relation_type, *_ = relation_tuple

                if relation_type not in (
                        NonEquivalenceRelationType.ArgumentOf,
                        NonEquivalenceRelationType.ReturnedValueOf
                ):
                    merged_out_edges_by_relation_tuple[relation_tuple].update(
                        (equivalent_node_disjoint_set.find(n) for n in out_edges)
                    )

        for relation_tuple, out_edges in merged_out_edges_by_relation_tuple.items():
            for from_node, to_node in itertools.pairwise(out_edges):
                equivalent_node_pair_queue.append((from_node, to_node))

    # ----------------------------------------------------------------------------------------------

    while equivalent_node_pair_queue or node_propagation_task_queue:
        # Unify equivalent nodes and induced equivalent nodes.
        while equivalent_node_pair_queue:
            first_node, second_node = equivalent_node_pair_queue.popleft()

            equivalent_node_disjoint_set.union(first_node, second_node)
            new_equivalence_relation_graph.set_equivalent(first_node, second_node)

        # Propagate runtime terms among equivalent sets.
        # Add induced equivalent relations.
        for top_node, disjoint_set_of_nodes in equivalent_node_disjoint_set.itersets():
            propagate_runtime_terms(disjoint_set_of_nodes)
            add_induced_equivalent_relations(disjoint_set_of_nodes)

        # Handle propagation tasks.
        while node_propagation_task_queue:
            node, propagation_task = node_propagation_task_queue.popleft()

            if propagation_task not in nodes_to_processed_propagation_task_sets[node]:
                if isinstance(propagation_task, AttributeAccessPropagationTask):
                    handle_attribute_access_propagation_task(node, propagation_task)
                elif isinstance(propagation_task, FunctionCallPropagationTask):
                    handle_function_call_propagation_task(node, propagation_task)

                nodes_to_processed_propagation_task_sets[node].add(propagation_task)

    # Merge the attribute counters, runtime term sets, non-equivalence relation of nodes which are equivalent.
    equivalent_node_disjoint_set_top_nodes_to_attribute_counters: defaultdict[ast.AST, AttributeCounter] = defaultdict(
        AttributeCounter)
    equivalent_node_disjoint_set_top_nodes_to_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]] = defaultdict(
        set)

    for node in new_nodes_to_attribute_counters.keys():
        equivalent_node_disjoint_set.find(node)

    for node in new_nodes_to_runtime_term_sets.keys():
        equivalent_node_disjoint_set.find(node)

    for top_node, disjoint_set_of_nodes in equivalent_node_disjoint_set.itersets():
        for node in disjoint_set_of_nodes:
            equivalent_node_disjoint_set_top_nodes_to_attribute_counters[top_node].update(
                new_nodes_to_attribute_counters[node])
            equivalent_node_disjoint_set_top_nodes_to_runtime_term_sets[top_node].update(
                new_nodes_to_runtime_term_sets[node])

    # ----------------------------------------------------------------------------------------------

    equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph = NonEquivalenceRelationGraph()

    for (from_, to, relation_tuple) in new_non_equivalence_relation_graph.iterate_relations():
        equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph.add_relation(
            equivalent_node_disjoint_set.find(from_),
            equivalent_node_disjoint_set.find(to),
            *relation_tuple
        )

    # ----------------------------------------------------------------------------------------------

    equivalent_node_disjoint_set_top_nodes_attribute_containment_graph: nx.DiGraph = nx.DiGraph()
    for from_, to in node_attribute_containment_graph.edges():
        equivalent_node_disjoint_set_top_nodes_attribute_containment_graph.add_edge(
            equivalent_node_disjoint_set.find(from_),
            equivalent_node_disjoint_set.find(to)
        )

    (
        condensed_equivalent_node_disjoint_set_top_nodes_attribute_containment_graph,
        strongly_connected_component_list,
        equivalent_node_disjoint_set_top_node_to_strongly_connected_component_index_dict
    ) = graph_condensation(
        equivalent_node_disjoint_set_top_nodes_attribute_containment_graph
    )

    strongly_connected_component_attribute_counter_list: list[AttributeCounter] = list()
    for strongly_connected_component in strongly_connected_component_list:
        strongly_connected_component_attribute_counter: AttributeCounter = AttributeCounter()
        for node in strongly_connected_component:
            strongly_connected_component_attribute_counter.update(
                equivalent_node_disjoint_set_top_nodes_to_attribute_counters[node]
            )
        strongly_connected_component_attribute_counter_list.append(strongly_connected_component_attribute_counter)

    propagated_strongly_connected_component_attribute_counter_list: list[AttributeCounter] = list()
    for strongly_connected_component_attribute_counter in strongly_connected_component_attribute_counter_list:
        propagated_strongly_connected_component_attribute_counter_list.append(
            strongly_connected_component_attribute_counter.copy()
        )

    for (
            first_strongly_connected_component_index,
            second_strongly_connected_component_index
    ) in topological_sort_edges(condensed_equivalent_node_disjoint_set_top_nodes_attribute_containment_graph):
        propagated_strongly_connected_component_attribute_counter_list[
            second_strongly_connected_component_index].update(
            propagated_strongly_connected_component_attribute_counter_list[first_strongly_connected_component_index]
        )

    for (
            strongly_connected_component,
            propagated_strongly_connected_component_attribute_counter
    ) in zip(
        strongly_connected_component_list,
        propagated_strongly_connected_component_attribute_counter_list
    ):
        for node in strongly_connected_component:
            equivalent_node_disjoint_set_top_nodes_to_attribute_counters[
                node] = propagated_strongly_connected_component_attribute_counter

    return (
        equivalent_node_disjoint_set,
        new_equivalence_relation_graph,
        new_non_equivalence_relation_graph,
        new_nodes_to_attribute_counters,
        equivalent_node_disjoint_set_top_nodes_to_attribute_counters,
        equivalent_node_disjoint_set_top_nodes_to_runtime_term_sets,
        equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph
    )
