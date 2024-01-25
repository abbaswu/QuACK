import _ast
import ast
import builtins
import collections.abc
import itertools
import logging
import typing
from collections import defaultdict

import debugging_singleton
import switches_singleton
from definitions_to_runtime_terms_mappings_singleton import top_level_class_definitions_to_runtime_classes, unwrapped_runtime_functions_to_named_function_definitions
from get_attributes_in_runtime_class import get_non_dynamic_attributes_in_runtime_class
from get_dict_for_runtime_class import get_dict_for_runtime_class
from get_parameters import get_parameters
from module_names_to_imported_names_to_runtime_objects_singleton import module_names_to_imported_names_to_runtime_objects
from parameter_lists_and_symbolic_return_values_singleton import nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values
from relations import NonEquivalenceRelationType
from typing_constraints_singleton import create_new_node, add_runtime_terms, set_node_to_be_instance_of, set_equivalent, \
    add_relation, create_related_node, update_attributes, \
    add_argument_of_returned_value_of_relations, add_containment_relation, add_two_way_containment_relation
from runtime_term import *
from node_visitor import *
from type_definitions import *
from unwrap import unwrap


unaryop_to_attribute: dict[type, str] = {
    ast.Invert: '__invert__',
    ast.UAdd: '__pos__',
    ast.USub: '__neg__'
}

operator_to_attribute: dict[type, str] = {
    ast.Add: '__add__',
    ast.Sub: '__sub__',
    ast.Mult: '__mul__',
    ast.MatMult: '__matmul__',
    ast.Div: '__truediv__',
    ast.Mod: '__mod__',
    ast.Pow: '__pow__',
    ast.LShift: '__lshift__',
    ast.RShift: '__rshift__',
    ast.BitOr: '__or__',
    ast.BitXor: '__xor__',
    ast.BitAnd: '__and__',
    ast.FloorDiv: '__floordiv__'
}

cmpop_to_attribute: dict[type, str] = {
    ast.Eq: '__eq__',
    ast.NotEq: '__ne__',
    ast.Lt: '__lt__',
    ast.LtE: '__le__',
    ast.Gt: '__gt__',
    ast.GtE: '__ge__'
}


async def collect_and_resolve_typing_constraints(
        module_names_to_module_nodes: typing.Mapping[str, ast.Module]
):
    """
    Collect and resolve typing constraints based on the semantics of each AST node.
    """
    # --------------------------------------------------------------------------------------------- #
    # For the unwrapped runtime functions and runtime classes,
    # And the literals True, False, Ellipsis, None, NotImplemented in builtins,
    # And each imported name within a module,
    # Initialize nodes which 'define' them,
    # And associate them with adequate runtime values.
    module_names_to_names_to_dummy_ast_nodes: defaultdict[str, dict[str, ast.AST]] = defaultdict(dict)

    names_to_nodes_for_builtins: dict[str, _ast.AST] = dict()

    for key, value in builtins.__dict__.items():
        if isinstance(value, (UnwrappedRuntimeFunction, RuntimeClass)):
            node = await create_new_node()

            names_to_nodes_for_builtins[key] = node
            await add_runtime_terms(node, {value})

    for value in (True, False, Ellipsis, None, NotImplemented):
        key = str(value)
        node = await create_new_node()

        names_to_nodes_for_builtins[key] = node
        await set_node_to_be_instance_of(node, type(value))

    for module_name, imported_names_to_runtime_objects in module_names_to_imported_names_to_runtime_objects.items():
        module_names_to_names_to_dummy_ast_nodes[module_name].update(names_to_nodes_for_builtins)

        for imported_name, runtime_object in imported_names_to_runtime_objects.items():
            node = await create_new_node()
            module_names_to_names_to_dummy_ast_nodes[module_name][imported_name] = node

            unwrapped_runtime_object = unwrap(runtime_object)
            runtime_term: RuntimeTerm | None = None

            if isinstance(unwrapped_runtime_object, Module):
                runtime_term = unwrapped_runtime_object
            if isinstance(unwrapped_runtime_object, RuntimeClass):
                runtime_term = unwrapped_runtime_object
            if isinstance(unwrapped_runtime_object, UnwrappedRuntimeFunction):
                if unwrapped_runtime_object in unwrapped_runtime_functions_to_named_function_definitions:
                    runtime_term = unwrapped_runtime_functions_to_named_function_definitions[unwrapped_runtime_object]
                else:
                    runtime_term = unwrapped_runtime_object

            if runtime_term is not None:
                logging.info(
                    'Matched imported name %s in module %s with unwrapped runtime object %s to runtime term %s',
                    imported_name, module_name, unwrapped_runtime_object, runtime_term
                )
                await add_runtime_terms(node, {runtime_term})

            else:
                logging.error(
                    'Cannot match imported name %s in module %s with unwrapped runtime object %s to a runtime term!',
                    imported_name, module_name, unwrapped_runtime_object
                )

    # --------------------------------------------------------------------------------------------- #
    # Update the runtime term sets of class definitions and function definitions.
    async def update_runtime_term_sets_callback(
            scope_stack: list[NodeProvidingScope],
            class_stack: list[ast.ClassDef],
            node: _ast.AST
    ):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            await add_runtime_terms(node, {node})
        # ast.ClassDef(name, bases, keywords, starargs, kwargs, body, decorator_list)
        if isinstance(node, ast.ClassDef):
            # Update the runtime term set of the current type variable.
            if node in top_level_class_definitions_to_runtime_classes:
                runtime_class = top_level_class_definitions_to_runtime_classes[node]
                await add_runtime_terms(node, {runtime_class})

    for module_name, module_node in module_names_to_module_nodes.items():
        await AsyncScopedNodeVisitor(update_runtime_term_sets_callback).visit(module_node)

    # --------------------------------------------------------------------------------------------- #
    # Set the default values of parameters to be equivalent to the corresponding type variables.
    async def handle_parameter_default_values_callback(
            scope_stack: list[NodeProvidingScope],
            class_stack: list[ast.ClassDef],
            node: _ast.AST
    ):
        # ast.FunctionDef(name, args, body, decorator_list, returns, type_comment)
        # ast.AsyncFunctionDef(name, args, body, decorator_list, returns, type_comment)
        # ast.Lambda(args, body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Get the parameter list of the current function.
            posargs, _, kwonlyargs, _ = get_parameters(node)

            # N posargs_defaults align with the *last* N posargs
            # N kw_defaults align with N kwonlyargs (though they may be None's)
            posargs_defaults = node.args.defaults
            kwonlyargs_defaults = node.args.kw_defaults

            for posarg, posarg_default in zip(
                reversed(posargs),
                reversed(posargs_defaults)
            ):
                await add_containment_relation(
                    superset_node=posarg,
                    subset_node=posarg_default
                )

            for kwonlyarg, kwonlyarg_default in zip(
                kwonlyargs,
                kwonlyargs_defaults
            ):
                if kwonlyarg_default is not None:
                    await add_containment_relation(
                        superset_node=kwonlyarg,
                        subset_node=kwonlyarg_default
                    )

    if switches_singleton.handle_parameter_default_values:
        for module_name, module_node in module_names_to_module_nodes.items():
            await AsyncScopedNodeVisitor(handle_parameter_default_values_callback).visit(module_node)

    # --------------------------------------------------------------------------------------------- #
    # Collect return value information for functions.
    # Resolve the (real) return value sets of nodes providing scope.
    nodes_providing_scope_to_apparent_return_value_sets: dict[NodeProvidingScope, set[ast.AST]] = dict()
    nodes_providing_scope_to_yield_value_sets: dict[NodeProvidingScope, set[ast.AST]] = dict()
    nodes_providing_scope_to_send_value_sets: dict[NodeProvidingScope, set[ast.AST]] = dict()
    nodes_providing_scope_returning_generators: set[NodeProvidingScope] = set()
    nodes_providing_scope_returning_coroutines: set[NodeProvidingScope] = set()

    async def collect_parameter_return_value_information_callback(
            scope_stack: list[NodeProvidingScope],
            class_stack: list[ast.ClassDef],
            node: _ast.AST
    ):
        # ast.FunctionDef(name, args, body, decorator_list, returns, type_comment)
        # ast.AsyncFunctionDef(name, args, body, decorator_list, returns, type_comment)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Initialize return type set, yield value set, send value set of the current scope.
            nodes_providing_scope_to_apparent_return_value_sets[node] = set()
            nodes_providing_scope_to_yield_value_sets[node] = set()
            nodes_providing_scope_to_send_value_sets[node] = set()

            if isinstance(node, ast.AsyncFunctionDef):
                nodes_providing_scope_returning_coroutines.add(node)
        # ast.Lambda(args, body)
        if isinstance(node, ast.Lambda):
            # Initialize return type set, yield value set, send value set of the current scope.
            nodes_providing_scope_to_apparent_return_value_sets[node] = {node.body}
            nodes_providing_scope_to_yield_value_sets[node] = set()
            nodes_providing_scope_to_send_value_sets[node] = set()
        # ast.Return(value)
        if isinstance(node, ast.Return):
            if scope_stack:
                current_scope = scope_stack[-1]

                if node.value is not None:
                    # Add the type variable of `value` to the return type set of the current scope.
                    nodes_providing_scope_to_apparent_return_value_sets[current_scope].add(node.value)
            else:
                logging.error('Cannot handle ast.Return outside of a scope!')
        # ast.Yield(value)
        if isinstance(node, ast.Yield):
            if scope_stack:
                current_scope = scope_stack[-1]

                nodes_providing_scope_returning_generators.add(current_scope)

                if node.value is not None:
                    # Add the type variable of `value` to the yield type set of the current scope.
                    nodes_providing_scope_to_yield_value_sets[current_scope].add(node.value)

                # Add the current type variable to the send type set of the current scope.
                nodes_providing_scope_to_send_value_sets[current_scope].add(node)
            else:
                logging.error('Cannot handle ast.Yield outside of a scope!')
        # ast.YieldFrom(value)
        if isinstance(node, ast.YieldFrom):
            if scope_stack:
                current_scope = scope_stack[-1]

                nodes_providing_scope_returning_generators.add(current_scope)

                # Add the $IterTargetOf$ the type variable of `value` to the yield type set of the current scope.
                nodes_providing_scope_to_yield_value_sets[current_scope].add(
                    await create_related_node(node.value, NonEquivalenceRelationType.IterTargetOf)
                )

                # Add the $SendTargetOf$ the type variable of `value` to the send type set of the current scope.
                nodes_providing_scope_to_send_value_sets[current_scope].add(
                    await create_related_node(node.value, NonEquivalenceRelationType.SendTargetOf)
                )
            else:
                logging.error('Cannot handle ast.YieldFrom outside of a scope!')

    for module_name, module_node in module_names_to_module_nodes.items():
        await AsyncScopedNodeVisitor(collect_parameter_return_value_information_callback).visit(module_node)

    nodes_providing_scope_set = set().union(
        nodes_providing_scope_to_apparent_return_value_sets.keys(),
        nodes_providing_scope_to_yield_value_sets.keys(),
        nodes_providing_scope_to_send_value_sets.keys()
    ) & nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values.keys()

    for node_providing_scope in nodes_providing_scope_set:
        _, _, symbolic_return_value = nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values[node_providing_scope]

        apparent_return_value_set = nodes_providing_scope_to_apparent_return_value_sets[node_providing_scope]
        yield_value_set = nodes_providing_scope_to_yield_value_sets[node_providing_scope]
        send_value_set = nodes_providing_scope_to_send_value_sets[node_providing_scope]

        # If there is no apparent return value, then add a dummy node to represent the return value of None.
        if not apparent_return_value_set:
            return_value: ast.AST = await create_new_node()
            await set_node_to_be_instance_of(return_value, type(None))

            augmented_apparent_return_value_set: set[_ast.AST] = {return_value}
        else:
            augmented_apparent_return_value_set: set[_ast.AST] = apparent_return_value_set.copy()

        # non-async functions returning generators
        if (
                node_providing_scope in nodes_providing_scope_returning_generators
                and node_providing_scope not in nodes_providing_scope_returning_coroutines
        ):
            await set_node_to_be_instance_of(symbolic_return_value, collections.abc.Generator)

            for yield_value in yield_value_set:
                await add_relation(symbolic_return_value, yield_value, NonEquivalenceRelationType.IterTargetOf)

            for send_value in send_value_set:
                await add_relation(symbolic_return_value, send_value, NonEquivalenceRelationType.SendTargetOf)

            for apparent_return_value in augmented_apparent_return_value_set:
                await add_relation(symbolic_return_value, apparent_return_value, NonEquivalenceRelationType.YieldFromAwaitResultOf)
        # async functions returning generators
        elif (
                node_providing_scope in nodes_providing_scope_returning_generators
                and node_providing_scope in nodes_providing_scope_returning_coroutines
        ):
            await set_node_to_be_instance_of(symbolic_return_value, collections.abc.AsyncGenerator)

            for yield_value in yield_value_set:
                await add_relation(symbolic_return_value, yield_value, NonEquivalenceRelationType.IterTargetOf)

            for send_value in send_value_set:
                await add_relation(symbolic_return_value, send_value, NonEquivalenceRelationType.SendTargetOf)
        # async functions not returning generators
        elif (
                 node_providing_scope not in nodes_providing_scope_returning_generators
                and node_providing_scope in nodes_providing_scope_returning_coroutines
        ):
            await set_node_to_be_instance_of(symbolic_return_value, collections.abc.Coroutine)

            for yield_value in yield_value_set:
                await add_relation(symbolic_return_value, yield_value, NonEquivalenceRelationType.IterTargetOf)

            for send_value in send_value_set:
                await add_relation(symbolic_return_value, send_value, NonEquivalenceRelationType.SendTargetOf)

            for apparent_return_value in augmented_apparent_return_value_set:
                await add_relation(symbolic_return_value, apparent_return_value, NonEquivalenceRelationType.YieldFromAwaitResultOf)
        # non-async functions not returning generators
        else:
            for apparent_return_value in augmented_apparent_return_value_set:
                await add_containment_relation(
                    superset_node=symbolic_return_value,
                    subset_node=apparent_return_value
                )

    # --------------------------------------------------------------------------------------------- #
    # The first parameter (`self`) of all instance methods within a runtime class are equivalent and are instances of the runtime class.
    # The first parameter (`cls`) of all classmethods within a runtime class contain the class definition as a runtime term.
    for top_level_class_definition, runtime_class in top_level_class_definitions_to_runtime_classes.items():
        first_parameter_of_instance_methods = set()
        first_parameter_of_classmethods = set()

        for k, v in get_dict_for_runtime_class(runtime_class).items():
            is_staticmethod = isinstance(v, staticmethod)
            is_classmethod = isinstance(v, classmethod)

            unwrapped_v = unwrap(v)

            if (
                    isinstance(unwrapped_v, UnwrappedRuntimeFunction)
                    and unwrapped_v in unwrapped_runtime_functions_to_named_function_definitions
            ):
                function_definition = unwrapped_runtime_functions_to_named_function_definitions[unwrapped_v]

                (
                    parameter_list,
                    parameter_name_to_parameter_mappings,
                    symbolic_return_value
                ) = nodes_to_parameter_lists_parameter_name_to_parameter_mappings_and_symbolic_return_values[function_definition]

                if parameter_list:
                    first_parameter: ast.arg = parameter_list[0]

                    if is_classmethod:
                        first_parameter_of_classmethods.add(first_parameter)
                    if not is_staticmethod and not is_classmethod:
                        first_parameter_of_instance_methods.add(first_parameter)

        # MULTI-WAY RELATION
        for (
                first_parameter_of_instance_method_1,
                first_parameter_of_instance_method_2
        ) in itertools.combinations(first_parameter_of_instance_methods, 2):
            await add_containment_relation(
                superset_node=first_parameter_of_instance_method_1,
                subset_node=first_parameter_of_instance_method_2
            )
            await add_containment_relation(
                superset_node=first_parameter_of_instance_method_2,
                subset_node=first_parameter_of_instance_method_1
            )

        for first_parameter_of_instance_method in first_parameter_of_instance_methods:
            await set_node_to_be_instance_of(first_parameter_of_instance_method, runtime_class)

        for first_parameter_of_classmethod in first_parameter_of_classmethods:
            await add_runtime_terms(first_parameter_of_classmethod, {runtime_class})

    # --------------------------------------------------------------------------------------------- #
    # Name resolution.
    # Resolve the names within each scope.
    def get_name_resolution_callback_function(module_name: str):
        # Keep track of what names are being defined at each scope.
        # None represents the global scope.
        nodes_providing_scope_to_local_names_to_definition_nodes: defaultdict[
            NodeProvidingScope | None, dict[str, ast.AST]
        ] = defaultdict(dict)
        nodes_providing_scope_to_local_names_to_definition_nodes[None].update(names_to_nodes_for_builtins)
        if module_name in module_names_to_names_to_dummy_ast_nodes:
            nodes_providing_scope_to_local_names_to_definition_nodes[None].update(
                module_names_to_names_to_dummy_ast_nodes[module_name]
            )

        nodes_providing_scope_to_explicit_global_names_to_definition_nodes: defaultdict[
            NodeProvidingScope | None, dict[str, ast.AST]
        ] = defaultdict(dict)

        nodes_providing_scope_to_explicit_nonlocal_names_to_definition_nodes: defaultdict[
            NodeProvidingScope | None, dict[str, ast.AST]
        ] = defaultdict(dict)

        async def handle_explicit_global_name_declaration(scope_stack: list[NodeProvidingScope], name: str) -> None:
            """
            Callback for encountered `ast.Global`'s.
            Adds the definition node of the name to (explicitly) global names within the current scope.
            """
            if scope_stack:
                current_scope = scope_stack[-1]

                # Find or create definition node within the global scope.
                if name in nodes_providing_scope_to_local_names_to_definition_nodes[None]:
                    # Directly retrieve the definition node
                    definition_node = nodes_providing_scope_to_local_names_to_definition_nodes[None][name]
                else:
                    # Add a dummy node as the definition node within the global scope.
                    definition_node = await create_new_node()
                    nodes_providing_scope_to_local_names_to_definition_nodes[None][name] = definition_node

                # Add the definition node to (explicitly) global names within the current scope.
                nodes_providing_scope_to_explicit_global_names_to_definition_nodes[current_scope][
                    name
                ] = definition_node
            else:
                logging.error('Cannot handle ast.Global nodes in the global scope!')

        async def handle_explicit_nonlocal_name_declaration(scope_stack: list[NodeProvidingScope], name: str) -> None:
            """
            Callback for encountered `ast.Nonlocal`'s.
            Adds the definition node of the name to (explicitly) global names within the current scope.
            """
            if scope_stack:
                current_scope = scope_stack[-1]

                # Find the name from parent scopes
                found_definition_node = False

                for scope in reversed(scope_stack[:-1]):
                    local_names_to_definition_nodes = nodes_providing_scope_to_local_names_to_definition_nodes[scope]
                    if name in local_names_to_definition_nodes:
                        # Directly retrieve the definition node
                        definition_node = local_names_to_definition_nodes[name]

                        # Add the definition node to (explicitly) nonlocal names within the current scope
                        nodes_providing_scope_to_explicit_nonlocal_names_to_definition_nodes[current_scope][
                            name] = definition_node

                        return

                if not found_definition_node:
                    logging.error(
                        'Cannot find the definition node of the nonlocal name %s given the scope stack %s!',
                        name, scope_stack
                    )
            else:
                logging.error('Cannot handle ast.Nonlocal nodes in the global scope!')

        async def get_last_definition_node(
                scope_stack: list[NodeProvidingScope],
                name: str,
                store: bool = False
        ) -> typing.Optional[ast.AST]:
            if scope_stack:
                current_scope = scope_stack[-1]
            else:
                current_scope = None

            last_definition_node: ast.AST | None = None

            # Is the name (explicitly) global within the current scope?
            if name in nodes_providing_scope_to_explicit_global_names_to_definition_nodes[current_scope]:
                # Directly retrieve the definition node
                last_definition_node = nodes_providing_scope_to_explicit_global_names_to_definition_nodes[current_scope][name]
            # Is the name (explicitly) nonlocal within the current scope?
            elif name in nodes_providing_scope_to_explicit_nonlocal_names_to_definition_nodes[current_scope]:
                # Directly retrieve the definition node
                last_definition_node = nodes_providing_scope_to_explicit_nonlocal_names_to_definition_nodes[current_scope][name]
            # Is the name local within the current scope?
            elif name in nodes_providing_scope_to_local_names_to_definition_nodes[current_scope]:
                # Directly retrieve the definition node
                last_definition_node = nodes_providing_scope_to_local_names_to_definition_nodes[current_scope][name]
            # The name may be (implicitly) global or nonlocal
            # In this case, the name is read
            elif not store:
                for containing_scope in itertools.chain(reversed(scope_stack[:-1]), (None,)):
                    local_names_to_definition_nodes = nodes_providing_scope_to_local_names_to_definition_nodes[
                        containing_scope]
                    if name in local_names_to_definition_nodes:
                        last_definition_node = local_names_to_definition_nodes[name]
                        break

            return last_definition_node

        async def handle_node_that_accesses_name(
                scope_stack: list[NodeProvidingScope],
                name: str,
                node: _ast.AST,
                store: bool = False
        ) -> _ast.AST:
            """
            Finds the last definition node for an accessed name under the current scope.
            If no definition node can be found,
            adds the node that accesses the name to local names within the current scope.
            """
            if scope_stack:
                current_scope = scope_stack[-1]
            else:
                current_scope = None

            last_definition_node: typing.Optional[ast.AST] = await get_last_definition_node(scope_stack, name, store)

            if last_definition_node is not None:
                logging.info(
                    'Found the last definition node %s for accesses name %s given the scope stack %s.',
                    last_definition_node, name, scope_stack
                )

                if store:
                    logging.info(
                        'We are storing, thus, we are redefining the name %s.',
                        name
                    )

                    nodes_providing_scope_to_local_names_to_definition_nodes[current_scope][name] = node
                    return node
                else:
                    return last_definition_node
            else:
                # Add the node that accesses the name to local names within the current scope
                if not store:
                    logging.error(
                        'Cannot find the last definition node for accessed name %s given the scope stack %s. Added a node that accesses the name to local names within the current scope.',
                        name,
                        scope_stack
                    )

                nodes_providing_scope_to_local_names_to_definition_nodes[current_scope][name] = node

                return node

        async def name_resolution_callback(
                scope_stack: list[NodeProvidingScope],
                class_stack: list[ast.ClassDef],
                node: _ast.AST
        ):
            # ast.Name(id, ctx)
            if isinstance(node, ast.Name):
                # Handle accessed name
                current_or_last_definition_node = await handle_node_that_accesses_name(
                    scope_stack,
                    node.id,
                    node,
                    isinstance(node.ctx, ast.Store)
                )

                if node != current_or_last_definition_node:
                    await add_two_way_containment_relation(
                        node,
                        current_or_last_definition_node
                    )

                    await set_equivalent({
                        node,
                        current_or_last_definition_node
                    }, True)
            # ast.AugAssign(target, op, value)
            if isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    last_definition_node: typing.Optional[ast.AST] = await get_last_definition_node(
                        scope_stack,
                        node.target.id,
                        False
                    )

                    if last_definition_node is not None:
                        await add_two_way_containment_relation(
                            node.target,
                            last_definition_node
                        )

                        await set_equivalent({
                            node.target,
                            last_definition_node,
                        }, True)
            # ast.ExceptHandler(type, name, body)
            if isinstance(node, ast.ExceptHandler):
                if node.name is not None:
                    # Handle accessed name
                    await handle_node_that_accesses_name(
                        scope_stack,
                        node.name,
                        node,
                        True
                    )
            # ast.FunctionDef(name, args, body, decorator_list, returns, type_comment)
            # ast.AsyncFunctionDef(name, args, body, decorator_list, returns, type_comment)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Handle accessed name.
                await handle_node_that_accesses_name(
                    scope_stack,
                    node.name,
                    node,
                    True
                )
            # ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults)
            if isinstance(node, ast.arguments):
                if scope_stack:
                    current_scope = scope_stack[-1]
                    for arg in node.posonlyargs + node.args:
                        # Handle accessed name
                        await handle_node_that_accesses_name(
                            scope_stack,
                            arg.arg,
                            arg,
                            True
                        )
                else:
                    logging.error('Cannot handle ast.arguments outside of a scope!')
            # ast.ClassDef(name, bases, keywords, starargs, kwargs, body, decorator_list)
            if isinstance(node, ast.ClassDef):
                # Handle accessed name.
                await handle_node_that_accesses_name(
                    scope_stack,
                    node.name,
                    node,
                    True
                )
            if isinstance(node, ast.Global):
                for name in node.names:
                    # Handle global name declaration
                    await handle_explicit_global_name_declaration(scope_stack, name)
            if isinstance(node, ast.Nonlocal):
                for name in node.names:
                    # Handle nonlocal name declaration
                    await handle_explicit_nonlocal_name_declaration(scope_stack, name)

        return name_resolution_callback

    for module_name, module_node in module_names_to_module_nodes.items():
        await AsyncScopedNodeVisitor(get_name_resolution_callback_function(module_name)).visit(module_node)

    # --------------------------------------------------------------------------------------------- #
    # Visit the AST nodes bottom-up.
    # Handle local syntax-directed typing constraints of each AST node.
    async def handle_local_syntax_directed_typing_constraints_callback_function(
            node: ast.AST
    ):
        # ast.Constant(value)
        if isinstance(node, ast.Constant):
            # Set the current type variable to be equivalent to `type(value)`
            await set_node_to_be_instance_of(node, type(node.value))

        # ast.JoinedStr(values)
        if isinstance(node, ast.JoinedStr):
            # Set the current type variable to be equivalent to `str`
            await set_node_to_be_instance_of(node, str)

        # ast.List(elts, ctx)
        if isinstance(node, ast.List):
            # Set the current type variable to be equivalent to `list`
            await set_node_to_be_instance_of(node, list)

            for elt in node.elts:
                if not isinstance(elt, ast.Starred):
                    # Set the type variable of `elt` as $ValueOf$ and $IterTargetOf$ the current type variable
                    await add_relation(node, elt, NonEquivalenceRelationType.ValueOf)
                    await add_relation(node, elt, NonEquivalenceRelationType.IterTargetOf)

            # Set $KeyOf$ the current type variable to be equivalent to `int`
            await set_node_to_be_instance_of(
                await create_related_node(node, NonEquivalenceRelationType.KeyOf),
                int
            )

        # ast.Tuple(elts, ctx)
        if isinstance(node, ast.Tuple):
            # Set the current type variable to be equivalent to `tuple`
            await set_node_to_be_instance_of(node, tuple)

            for i, elt in enumerate(node.elts):
                if not isinstance(elt, ast.Starred):
                    # Set the type variable of `elt` as the $i$-th $ElementOf$ the current type variable
                    await add_relation(node, elt, NonEquivalenceRelationType.ElementOf, i)
                else:
                    break

            # Set $KeyOf$ the current type variable to be equivalent to `int`
            await set_node_to_be_instance_of(
                await create_related_node(node, NonEquivalenceRelationType.KeyOf),
                int
            )

        # ast.Set(elts)
        if isinstance(node, ast.Set):
            # Set the current type variable to be equivalent to `set`
            await set_node_to_be_instance_of(node, set)

            for elt in node.elts:
                if not isinstance(elt, ast.Starred):
                    # Set the type variable of `elt` as $IterTargetOf$ the current type variable
                    await add_relation(node, elt, NonEquivalenceRelationType.IterTargetOf)

        # ast.Dict(keys, values)
        if isinstance(node, ast.Dict):
            # Set the current type variable to be equivalent to `dict`
            await set_node_to_be_instance_of(node, dict)

            for key_, value_ in zip(node.keys, node.values):
                if key_ is not None:
                    # Set the type variable of `key` as $KeyOf$ and $IterTargetOf$ the current type variable
                    await add_relation(node, key_, NonEquivalenceRelationType.KeyOf)
                    await add_relation(node, key_, NonEquivalenceRelationType.IterTargetOf)
                    # Set the type variable of `value` as $ValueOf$ the current type variable
                    await add_relation(node, value_, NonEquivalenceRelationType.ValueOf)
                else:
                    # as described in https://docs.python.org/3/reference/expressions.html#dictionary-displays
                    # Set the type variable of `value` to be equivalent to `collections.abc.Mapping`
                    await set_node_to_be_instance_of(value_, collections.abc.Mapping)

                    # Set the $KeyOf$, $ValueOf$, and $IterTargetOf$ the type variable of `value` as equivalent to the $KeyOf$, $ValueOf$, and $IterTargetOf$ the current type variable.
                    await add_two_way_containment_relation(
                        await create_related_node(node, NonEquivalenceRelationType.KeyOf),
                        await create_related_node(value_, NonEquivalenceRelationType.KeyOf)
                    )

                    await add_two_way_containment_relation(
                        await create_related_node(node, NonEquivalenceRelationType.ValueOf),
                        await create_related_node(value_, NonEquivalenceRelationType.ValueOf)
                    )

                    # TWO-WAY RELATION
                    await add_two_way_containment_relation(
                        await create_related_node(node, NonEquivalenceRelationType.IterTargetOf),
                        await create_related_node(value_, NonEquivalenceRelationType.IterTargetOf)
                    )

        # ast.Starred(value, ctx)
        if isinstance(node, ast.Starred):
            # Set the type variable of `value` to be equivalent to `collections.abc.Iterable`
            # according to https://docs.python.org/3/reference/expressions.html#grammar-token-python-grammar-starred_expression)
            await set_node_to_be_instance_of(node.value, collections.abc.Iterable)

        # ast.UnaryOp(op, operand)
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.Not):
                # Update the attribute counter of the type variable of `operand` with the attribute corresponding to `op`.
                await update_attributes(node.operand, {unaryop_to_attribute[type(node.op)]})

                # Set the current type variable as equivalent to the type variable of `operand`.
                await add_two_way_containment_relation(
                    node,
                    node.operand
                )
            else:
                # Set the current type variable as equivalent to `bool`.
                await set_node_to_be_instance_of(node, bool)

        # ast.BinOp(left, op, right)
        if isinstance(node, ast.BinOp):
            # Update the attribute counter of the type variable of `left` with the attribute corresponding to `op`.
            await update_attributes(node.left, {operator_to_attribute[type(node.op)]})

            # Set the current type variable as equivalent to the type variable of `left`.
            await add_two_way_containment_relation(
                node,
                node.left
            )

            if not isinstance(node, (ast.Mod, ast.Mult)):
                await add_two_way_containment_relation(
                    node.left,
                    node.right
                )

        # ast.Compare(left, ops, comparators)
        if isinstance(node, ast.Compare):
            operands = [node.left] + node.comparators
            for (left, right), op in zip(
                    itertools.pairwise(operands),
                    node.ops
            ):
                if isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    # Update the attribute counter of the type variable of `left` with the attribute corresponding to `op`.
                    await update_attributes(left, {cmpop_to_attribute[type(op)]})

                    # Set the type variable of `left` as equivalent to the type variable of `right`.
                    await add_two_way_containment_relation(
                        left,
                        right
                    )
                elif isinstance(op, (ast.In, ast.NotIn)):
                    # based on https://docs.python.org/3/reference/expressions.html#membership-test-operations and https://discuss.python.org/t/deprecate-old-style-iteration-protocol/17863/7
                    # Update the attribute counter of the type variable of `right` with the attributes `__contains__` and `__iter__`.
                    await update_attributes(right, {'__contains__', '__iter__'})

                    # Set the type variable of `left` as $IterTargetOf$ the type variable of `right`.
                    await add_relation(right, left, NonEquivalenceRelationType.IterTargetOf)

                    # Set the current type variable as equivalent to `bool`.
                    await set_node_to_be_instance_of(node, bool)

        # ast.Call(func, args, keywords, starargs, kwargs)
        if isinstance(node, ast.Call):
            # Update the attribute counter of the type variable of `func` with the attribute `__call__`.
            await update_attributes(node.func, {'__call__'})

            undetermined_number_of_parameters: bool = False

            argument_node_set_list: list[set[_ast.AST]] = []

            for i, arg in enumerate(node.args):
                if not isinstance(arg, ast.Starred):
                    argument_node_set_list.append({arg})
                else:
                    undetermined_number_of_parameters = True
                    break

            for keyword in node.keywords:
                if keyword.arg is None:
                    # Set the type variable of `keyword.value` as equivalent to `collections.abc.Mapping`.
                    # as described in https://docs.python.org/3/reference/expressions.html#dictionary-displays
                    await set_node_to_be_instance_of(keyword.value, collections.abc.Mapping)

                    # Set the $KeyOf$ the type variable of `keyword.value` as equivalent to `str`.
                    await set_node_to_be_instance_of(
                        await create_related_node(keyword.value, NonEquivalenceRelationType.KeyOf),
                        str
                    )

            if node.keywords:
                undetermined_number_of_parameters = True

            if undetermined_number_of_parameters:
                # Create a dummy node to represent all parameters.
                dummy_node_representing_all_parameters = await create_new_node()

                await set_node_to_be_instance_of(dummy_node_representing_all_parameters, type(Ellipsis))

                if argument_node_set_list:
                    argument_node_set_list[0].add(dummy_node_representing_all_parameters)
                else:
                    argument_node_set_list.append({dummy_node_representing_all_parameters})

            # Set the type variable of `arg` as the $i$-th $ParameterOf$ the type variable of `func`.
            # Set the current type variable as the $ReturnValueOf$ the type variable of `func`.
            await add_argument_of_returned_value_of_relations(
                node.func,
                argument_node_set_list,
                {node}
            )

        # ast.IfExp(test, body, orelse)
        if isinstance(node, ast.IfExp):
            await add_containment_relation(
                superset_node=node,
                subset_node=node.body
            )

            await add_containment_relation(
                superset_node=node,
                subset_node=node.orelse
            )

        # ast.Attribute(value, attr, ctx)
        if isinstance(node, ast.Attribute):
            # Update the attribute counter of the type variable of `value` with `attr`.
            await update_attributes(node.value, {node.attr})

            # Set the current type variable as the $attr$-$AttrOf$ the type variable of `value`.
            await add_relation(node.value, node, NonEquivalenceRelationType.AttrOf, node.attr)

        # ast.NamedExpr(target, value)
        if isinstance(node, ast.NamedExpr):
            # Set the current type variable as equivalent to the type variable of `target` and `value`.
            await add_two_way_containment_relation(
                node.target,
                node.value
            )

            await add_two_way_containment_relation(
                node,
                node.target
            )

            await set_equivalent({
                node,
                node.target,
                node.value
            }, True)

        # ast.Subscript(value, slice, ctx)
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, (ast.Tuple, ast.Slice)):
                # Set the current type variable as equivalent to the type variable of `value`.
                await add_two_way_containment_relation(
                    node,
                    node.value
                )
            else:
                # Set the current type variable as $ValueOf$ the type variable of `value`.
                await add_relation(node.value, node, NonEquivalenceRelationType.ValueOf)

                # Set the type variable of `slice` as $KeyOf$ the type variable of `value`.
                await add_relation(node.value, node.slice, NonEquivalenceRelationType.KeyOf)

            if isinstance(node.ctx, ast.Load):
                # Update the attribute counter of the type variable of `value` with the attribute `__getitem__`.
                await update_attributes(node.value, {'__getitem__'})

            if isinstance(node.ctx, ast.Store):
                # Update the attribute counter of the type variable of `value` with the attribute `__setitem__`.
                await update_attributes(node.value, {'__setitem__'})

        # ast.Slice(lower, upper, step)
        if isinstance(node, ast.Slice):
            # Set the current type variable as equivalent to `slice`.
            await set_node_to_be_instance_of(node, slice)

            for value in (node.lower, node.upper, node.step):
                if value is not None:
                    # Set the type variable of `value` as equivalent to `int`.
                    await set_node_to_be_instance_of(value, int)

        # ast.ListComp(elt, generators)
        if isinstance(node, ast.ListComp):
            # Set the current type variable as equivalent to `list`.
            await set_node_to_be_instance_of(node, list)

            # Set the type variable of `elt` as $ValueOf$ and $IterTargetOf$ the current type variable.
            await add_relation(node, node.elt, NonEquivalenceRelationType.ValueOf)
            await add_relation(node, node.elt, NonEquivalenceRelationType.IterTargetOf)

            # Set $KeyOf$ the current type variable as equivalent to `int`.
            await set_node_to_be_instance_of(
                await create_related_node(node, NonEquivalenceRelationType.KeyOf),
                int
            )

        # ast.SetComp(elt, generators)
        if isinstance(node, ast.SetComp):
            # Set the current type variable as equivalent to `set`.
            await set_node_to_be_instance_of(node, set)

            # Set the type variable of `elt` as $IterTargetOf$ the current type variable.
            await add_relation(node, node.elt, NonEquivalenceRelationType.IterTargetOf)

        # ast.GeneratorExp(elt, generators)
        if isinstance(node, ast.GeneratorExp):
            # Set the current type variable as equivalent to `collections.abc.Generator`.
            await set_node_to_be_instance_of(node, collections.abc.Generator)

            # Set the type variable of `elt` as $IterTargetOf$ the current type variable.
            await add_relation(node, node.elt, NonEquivalenceRelationType.IterTargetOf)

        # ast.DictComp(key, value, generators)
        if isinstance(node, ast.DictComp):
            # Set the current type variable as equivalent to `dict`.
            await set_node_to_be_instance_of(node, dict)

            # Set the type variable of `key` as $KeyOf$ and $IterTargetOf$ the current type variable.
            await add_relation(node, node.key, NonEquivalenceRelationType.KeyOf)
            await add_relation(node, node.key, NonEquivalenceRelationType.IterTargetOf)

            # Set the type variable of `value` as $ValueOf$ the current type variable.
            await add_relation(node, node.value, NonEquivalenceRelationType.ValueOf)

        # ast.comprehension(target, iter, ifs, is_async)
        if isinstance(node, ast.comprehension):
            if node.is_async:
                # Update the attribute counter of the type variable of `iter` with the attribute `__aiter__`.
                await update_attributes(node.iter, {'__aiter__'})
            else:
                # Update the attribute counter of the type variable of `iter` with the attribute `__iter__`.
                await update_attributes(node.iter, {'__iter__'})

            # Set the type variable of `target` as $IterTargetOf$ the type variable of `iter`.
            await add_relation(node.iter, node.target, NonEquivalenceRelationType.IterTargetOf)

        # ast.Assign(targets, value, type_comment)
        if isinstance(node, ast.Assign):
            async def _r(
                    targets: list[ast.expr],
                    value: ast.expr
            ):
                if targets:
                    targets_up_front, last_target = targets[:-1], targets[-1]

                    await add_two_way_containment_relation(
                        last_target,
                        value
                    )

                    await _r(
                        targets_up_front,
                        last_target
                    )

            await _r(
                node.targets,
                node.value
            )

            await set_equivalent({
                node.value,
                *node.targets
            }, True)

        # ast.AnnAssign(target, annotation, value, simple)
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                await add_two_way_containment_relation(
                    node.target,
                    node.value
                )

                await set_equivalent({
                    node.target,
                    node.value
                }, True)

        # ast.AugAssign(target, op, value)
        if isinstance(node, ast.AugAssign):
            # Update the attribute counter of the type variable of `target` with the attribute corresponding to `op`.
            await update_attributes(node.target, {operator_to_attribute[type(node.op)]})

            await add_two_way_containment_relation(
                node.target,
                node.value
            )

        # ast.For(target, iter, body, orelse, type_comment)
        if isinstance(node, ast.For):
            # Update the attribute counter of the type variable of `iter` with the attribute `__iter__`.
            await update_attributes(node.iter, {'__iter__'})

            # Set the type variable of `target` as $IterTargetOf$ the type variable of `iter`.
            await add_relation(node.iter, node.target, NonEquivalenceRelationType.IterTargetOf)

        # ast.AsyncFor(target, iter, body, orelse, type_comment)
        if isinstance(node, ast.AsyncFor):
            # Update the attribute counter of the type variable of `iter` with the attribute `__aiter__`.
            await update_attributes(node.iter, {'__aiter__'})

            # Set the type variable of `target` as $IterTargetOf$ the type variable of `iter`.
            await add_relation(node.iter, node.target, NonEquivalenceRelationType.IterTargetOf)

        # ast.With(items, body, type_comment)
        if isinstance(node, ast.With):
            for withitem in node.items:
                # Update the attribute counter of the type variable of `withitem.context_expr` with the attributes `__enter__`, `__exit__`.
                await update_attributes(withitem.context_expr, {'__enter__', '__exit__'})

                if withitem.optional_vars is not None:
                    # `getattr_node = ast.Attribute(value=withitem.context_expr, attr='__enter__', ctx=ast.Load())`
                    getattr_node = await create_new_node()

                    await add_relation(
                        withitem.context_expr,
                        getattr_node,
                        NonEquivalenceRelationType.AttrOf,
                        '__enter__'
                    )

                    # Set the type variable of `withitem.optional_vars` as the $ReturnValueOf$ the type variable of `getattr_node`.
                    await add_relation(
                        getattr_node,
                        withitem.optional_vars,
                        NonEquivalenceRelationType.ReturnedValueOf
                    )

        # ast.AsyncWith(items, body, type_comment)
        if isinstance(node, ast.AsyncWith):
            for withitem in node.items:
                # Update the attribute counter of the type variable of `withitem.context_expr` with the attributes `__aenter__`, `__aexit__`.
                await update_attributes(withitem.context_expr, {'__aenter__', '__aexit__'})

                if withitem.optional_vars is not None:
                    # `getattr_node = ast.Attribute(value=withitem.context_expr, attr='__aenter__', ctx=ast.Load())`
                    getattr_node = await create_new_node()

                    await add_relation(
                        withitem.context_expr,
                        getattr_node,
                        NonEquivalenceRelationType.AttrOf,
                        '__aenter__'
                    )

                    # Set the type variable of `withitem.optional_vars` as the $YieldFromAwaitResultOf$ the $ReturnValueOf$ type variable of `getattr_node`.
                    await add_relation(
                        await create_related_node(getattr_node, NonEquivalenceRelationType.ReturnedValueOf),
                        withitem.optional_vars,
                        NonEquivalenceRelationType.YieldFromAwaitResultOf
                    )

        # ast.YieldFrom(value)
        if isinstance(node, ast.YieldFrom):
            # Set the current type variable as the $YieldFromAwaitResultOf$ the type variable of `value`.
            await add_relation(node.value, node, NonEquivalenceRelationType.YieldFromAwaitResultOf)

            # Update the attribute counter of the type variable of `value` with the attribute `__iter__`.
            await update_attributes(node.value, {'__iter__'})
        # ast.Await(value)
        if isinstance(node, ast.Await):
            # Update the attribute counter of the type variable of `value` with the attribute `__await__`.
            await update_attributes(node.value, {'__await__'})

            # Set the current type variable as the $YieldFromAwaitResultOf$ of the type variable of `value`
            await add_relation(node.value, node, NonEquivalenceRelationType.YieldFromAwaitResultOf)

    for module_name, module_node in module_names_to_module_nodes.items():
        await AsyncEvaluationOrderBottomUpNodeVisitor(handle_local_syntax_directed_typing_constraints_callback_function).visit(module_node)
