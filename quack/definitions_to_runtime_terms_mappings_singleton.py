"""
NOTE: This module is intended to be used as a singleton.
Initialize it by calling get_definitions_to_runtime_terms_mappings() once.
"""

import logging

from get_dict_for_runtime_class import get_comprehensive_dict_for_runtime_class
from node_visitor import *
from type_definitions import *
from unwrap import unwrap

top_level_class_definitions_to_runtime_classes: dict[ast.ClassDef, RuntimeClass] = dict()

unwrapped_runtime_functions_to_named_function_definitions: dict[
    UnwrappedRuntimeFunction,
    NamedFunctionDefinition
] = dict()


def get_definitions_to_runtime_terms_mappings(
        module_names: typing.Iterable[str],
        modules: typing.Iterable[types.ModuleType],
        module_nodes: typing.Iterable[ast.Module]
):
    """
    Try to map top-level class definitions to runtime classes (helps when initializing their runtime terms)
    and unwrapped runtime functions to named function definitions (helps when doing runtime term propagation).
    """
    # The real name tuples of runtime classes
    real_name_tuples_of_runtime_classes: dict[tuple[str, ...], RuntimeClass] = dict()
    # The real name tuples of unwrapped runtime functions
    real_name_tuples_of_unwrapped_runtime_functions: dict[tuple[str, ...], UnwrappedRuntimeFunction] = dict()

    # The real name tuples of class definitions
    real_name_tuples_of_top_level_class_definitions: dict[tuple[str, ...], ast.ClassDef] = dict()
    # The real name tuples of named function definitions
    real_name_tuples_of_named_function_definitions: dict[tuple[str, ...], NamedFunctionDefinition] = dict()

    def get_scoped_node_visitor_callback(module_name: str):
        def scoped_node_visitor_callback(
                scope_stack: list[NodeProvidingScope],
                class_stack: list[ast.ClassDef],
                node: ast.AST
        ):
            nonlocal real_name_tuples_of_top_level_class_definitions, real_name_tuples_of_named_function_definitions

            # Top-level class definition
            if isinstance(node, ast.ClassDef) and not scope_stack and not class_stack:
                real_name_tuples_of_top_level_class_definitions[(module_name, node.name)] = node
            # Named function definition
            if isinstance(node, NamedFunctionDefinition) and not scope_stack:
                # Top-level named function definition
                if not class_stack:
                    real_name_tuples_of_named_function_definitions[(module_name, node.name)] = node
                # Named function node within top-level class node
                elif len(class_stack) == 1 and isinstance(class_node := class_stack[0], ast.ClassDef):
                    real_name_tuples_of_named_function_definitions[(module_name, class_node.name, node.name)] = node

        return scoped_node_visitor_callback

    for module_name, module, module_node in zip(module_names, modules, module_nodes):
        # These may be either imported or defined
        for k, v in module.__dict__.items():
            unwrapped_v = unwrap(v)

            if isinstance(unwrapped_v, RuntimeClass):
                real_module_name = unwrapped_v.__module__
                real_class_name = unwrapped_v.__name__
                real_name_tuples_of_runtime_classes[(real_module_name, real_class_name)] = unwrapped_v

                for k_, v_ in get_comprehensive_dict_for_runtime_class(unwrapped_v).items():
                    unwrapped_v_ = unwrap(v_)
                    if isinstance(unwrapped_v_, UnwrappedRuntimeFunction):
                        real_name_tuples_of_unwrapped_runtime_functions[
                            (real_module_name, real_class_name, unwrapped_v_.__name__)] = unwrapped_v_

            if isinstance(unwrapped_v, UnwrappedRuntimeFunction):
                real_name_tuples_of_unwrapped_runtime_functions[
                    (unwrapped_v.__module__, unwrapped_v.__name__)] = unwrapped_v

        ScopedNodeVisitor(get_scoped_node_visitor_callback(module_name)).visit(module_node)

    for real_name_tuple, top_level_class_definition in real_name_tuples_of_top_level_class_definitions.items():
        if real_name_tuple in real_name_tuples_of_runtime_classes:
            runtime_class = real_name_tuples_of_runtime_classes[real_name_tuple]

            top_level_class_definitions_to_runtime_classes[top_level_class_definition] = runtime_class
            logging.info('Matched class definition %s to runtime class %s', '.'.join(real_name_tuple), runtime_class)
        else:
            logging.error('Cannot match class definition %s to a runtime class!', '.'.join(real_name_tuple))

    for real_name_tuple, named_function_definition in real_name_tuples_of_named_function_definitions.items():
        if real_name_tuple in real_name_tuples_of_unwrapped_runtime_functions:
            unwrapped_runtime_function = real_name_tuples_of_unwrapped_runtime_functions[real_name_tuple]

            unwrapped_runtime_functions_to_named_function_definitions[
                unwrapped_runtime_function] = named_function_definition
            logging.info('Matched unwrapped runtime function %s to named function definition %s', '.'.join(real_name_tuple), ast.unparse(named_function_definition))
        else:
            logging.error('Cannot match unwrapped runtime function %s to a named function definition!', '.'.join(real_name_tuple))
