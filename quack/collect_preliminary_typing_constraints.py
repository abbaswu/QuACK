import ast
import builtins
import collections.abc
import itertools
import logging
from collections import defaultdict

from attribute_counter import AttributeCounter
from get_attributes_in_runtime_class import get_attributes_in_runtime_class, get_non_dynamic_attributes_in_runtime_class
from get_dict_for_runtime_class import get_comprehensive_dict_for_runtime_class, get_dict_for_runtime_class
from get_parameters import get_parameters
from relations import NonEquivalenceRelationType, EquivalenceRelationGraph, NonEquivalenceRelationGraph
from runtime_term import *
from scoped_node_visitor import *
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


def collect_preliminary_typing_constraints(
        top_level_class_definitions_to_runtime_classes: typing.Mapping[ast.ClassDef, RuntimeClass],
        unwrapped_runtime_functions_to_named_function_definitions: typing.Mapping[
            UnwrappedRuntimeFunction, NamedFunctionDefinition],
        module_names_to_imported_names_to_runtime_objects: typing.Mapping[str, typing.Mapping[str, object]],
        module_names_to_module_nodes: typing.Mapping[str, ast.Module]
):
    """
    Collect preliminary typing constraints based on the semantics of each AST node.
    """
    nodes_to_attribute_counters: defaultdict[ast.AST, AttributeCounter] = defaultdict(AttributeCounter)
    nodes_to_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]] = defaultdict(set)
    nodes_providing_scope_to_parameter_lists: defaultdict[NodeProvidingScope, list[ast.arg]] = defaultdict(list)
    nodes_providing_scope_to_apparent_return_value_sets: defaultdict[NodeProvidingScope, set[ast.AST]] = defaultdict(
        set)
    nodes_providing_scope_to_yield_value_sets: defaultdict[NodeProvidingScope, set[ast.AST]] = defaultdict(set)
    nodes_providing_scope_to_send_value_sets: defaultdict[NodeProvidingScope, set[ast.AST]] = defaultdict(set)
    nodes_providing_scope_returning_generators: set[NodeProvidingScope] = set()
    nodes_providing_scope_returning_coroutines: set[NodeProvidingScope] = set()
    equivalence_relation_graph: EquivalenceRelationGraph = EquivalenceRelationGraph()
    non_equivalence_relation_graph: NonEquivalenceRelationGraph = NonEquivalenceRelationGraph()

    def add_new_dummy_node() -> ast.AST:
        dummy_node = ast.AST()

        nodes_to_attribute_counters[dummy_node] = AttributeCounter()
        nodes_to_runtime_term_sets[dummy_node] = set()
        equivalence_relation_graph.add_node(dummy_node)
        non_equivalence_relation_graph.add_node(dummy_node)

        return dummy_node

    def set_node_to_be_runtime_class(
            node: ast.AST,
            runtime_class: RuntimeClass
    ):
        # Update attribute counter
        nodes_to_attribute_counters[node].update(get_non_dynamic_attributes_in_runtime_class(runtime_class))

        # Update runtime term set
        nodes_to_runtime_term_sets[node].add(runtime_class)

    def set_node_to_be_instance_of(
            node: ast.AST,
            runtime_class: RuntimeClass
    ):
        # Update attribute counter
        if runtime_class not in (
                type(None),
                type(Ellipsis),
                type(NotImplemented),
        ):
            nodes_to_attribute_counters[node].update(get_attributes_in_runtime_class(runtime_class))

        # Update runtime term set
        nodes_to_runtime_term_sets[node].add(Instance(runtime_class))

    # For the unwrapped runtime functions and runtime classes
    # And the literals True, False, Ellipsis, None, NotImplemented in builtins
    # Initialize dummy nodes which 'define' them
    # And associate them with adequate runtime values
    names_to_dummy_nodes_for_builtins: dict[str, ast.AST] = dict()

    for key, value in builtins.__dict__.items():
        if isinstance(value, (UnwrappedRuntimeFunction, RuntimeClass)):
            dummy_node = add_new_dummy_node()

            names_to_dummy_nodes_for_builtins[key] = dummy_node
            nodes_to_runtime_term_sets[dummy_node].add(value)

    for value in (True, False, Ellipsis, None, NotImplemented):
        key = str(value)
        dummy_node = add_new_dummy_node()

        names_to_dummy_nodes_for_builtins[key] = dummy_node
        set_node_to_be_instance_of(dummy_node, type(value))

    # For each imported name within a module
    # Initialize dummy nodes which 'define' them
    # And associate them with adequate runtime values
    module_names_to_imported_names_to_dummy_ast_nodes: defaultdict[str, dict[str, ast.AST]] = defaultdict(dict)
    for module_name, imported_names_to_runtime_objects in module_names_to_imported_names_to_runtime_objects.items():
        for imported_name, runtime_object in imported_names_to_runtime_objects.items():
            dummy_node = add_new_dummy_node()
            module_names_to_imported_names_to_dummy_ast_nodes[module_name][imported_name] = dummy_node

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
                nodes_to_runtime_term_sets[dummy_node].add(runtime_term)
                logging.info(
                    'Matched imported name %s in module %s with unwrapped runtime object %s to runtime term %s',
                    imported_name, module_name, unwrapped_runtime_object, runtime_term)
            else:
                logging.error(
                    'Cannot match imported name %s in module %s with unwrapped runtime object %s to a runtime term!',
                    imported_name, module_name, unwrapped_runtime_object)

    # Define the callback function for ScopedNodeVisitor, executed on each module node
    def get_scoped_node_visitor_callback_function(module_name: str):
        # Keep track of what names are being defined at each scope
        # None represents the global scope
        nodes_providing_scope_to_local_names_to_definition_nodes: defaultdict[
            NodeProvidingScope | None, dict[str, ast.AST]] = defaultdict(dict)
        nodes_providing_scope_to_explicit_global_names_to_definition_nodes: defaultdict[
            NodeProvidingScope | None, dict[str, ast.AST]] = defaultdict(dict)
        nodes_providing_scope_to_explicit_nonlocal_names_to_definition_nodes: defaultdict[
            NodeProvidingScope | None, dict[str, ast.AST]] = defaultdict(dict)

        nodes_providing_scope_to_local_names_to_definition_nodes[None].update(names_to_dummy_nodes_for_builtins)
        logging.info('Loaded dummy AST nodes for builtins in module %s', module_name)

        if module_name in module_names_to_imported_names_to_dummy_ast_nodes:
            nodes_providing_scope_to_local_names_to_definition_nodes[None].update(
                module_names_to_imported_names_to_dummy_ast_nodes[module_name])
            logging.info('Loaded dummy AST nodes for imported names in module %s', module_name)

        def handle_explicit_global_name_declaration(scope_stack: list[NodeProvidingScope], name: str) -> None:
            """
            Callback for encountered `ast.Global`'s.
            Adds the definition node of the name to (explicitly) global names within the current scope.
            """
            nonlocal nodes_providing_scope_to_local_names_to_definition_nodes, nodes_providing_scope_to_explicit_global_names_to_definition_nodes

            if scope_stack:
                current_scope = scope_stack[-1]

                # Find or create definition node within the global scope.
                if name in nodes_providing_scope_to_local_names_to_definition_nodes[None]:
                    # Directly retrieve the definition node
                    definition_node = nodes_providing_scope_to_local_names_to_definition_nodes[None][name]
                else:
                    # Add a dummy node as the definition node within the global scope.
                    definition_node = add_new_dummy_node()
                    nodes_providing_scope_to_local_names_to_definition_nodes[None][name] = definition_node

                # Add the definition node to (explicitly) global names within the current scope.
                nodes_providing_scope_to_explicit_global_names_to_definition_nodes[current_scope][
                    name] = definition_node
            else:
                logging.error('Cannot handle ast.Global nodes in the global scope!')

        def handle_explicit_nonlocal_name_declaration(scope_stack: list[NodeProvidingScope], name: str) -> None:
            """
            Callback for encountered `ast.Nonlocal`'s.
            Adds the definition node of the name to (explicitly) global names within the current scope.
            """
            nonlocal nodes_providing_scope_to_local_names_to_definition_nodes, nodes_providing_scope_to_explicit_nonlocal_names_to_definition_nodes

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
                    logging.error('Cannot find the definition node of the nonlocal name %s given the scope stack %s!',
                                  name, scope_stack)
            else:
                logging.error('Cannot handle ast.Nonlocal nodes in the global scope!')

        def get_last_definition_node(
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
                last_definition_node = nodes_providing_scope_to_explicit_global_names_to_definition_nodes[current_scope][
                    name]
            # Is the name (explicitly) nonlocal within the current scope?
            elif name in nodes_providing_scope_to_explicit_nonlocal_names_to_definition_nodes[current_scope]:
                # Directly retrieve the definition node
                last_definition_node = nodes_providing_scope_to_explicit_nonlocal_names_to_definition_nodes[current_scope][
                    name]
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

        def handle_node_that_accesses_name(
                scope_stack: list[NodeProvidingScope],
                name: str,
                node: ast.AST,
                store: bool = False
        ) -> None:
            """
            Finds the last definition node for an accessed name under the current scope,
            and optionally sets the last definition node as equivalent to the node that accesses the name.
            If no definition node can be found,
            adds the node that accesses the name to local names within the current scope.
            """
            if scope_stack:
                current_scope = scope_stack[-1]
            else:
                current_scope = None

            last_definition_node: typing.Optional[ast.AST] = get_last_definition_node(scope_stack, name, store)

            if last_definition_node is not None:
                logging.info('Found the last definition node %s for accesses name %s given the scope stack %s.',
                             last_definition_node, name, scope_stack)

                if store:
                    nodes_providing_scope_to_local_names_to_definition_nodes[current_scope][name] = node
                    logging.info(
                        'We are storing, thus, we are redefining the name %s.',
                        name
                    )
                else:
                    # Set the last definition node to be equivalent to the current node
                    equivalence_relation_graph.set_equivalent(node, last_definition_node)
                    logging.info(
                        'We are loading, thus, we set the last definition node to be equivalent to the current node.'
                    )
            else:
                # Add the node that accesses the name to local names within the current scope
                nodes_providing_scope_to_local_names_to_definition_nodes[current_scope][name] = node

                if not store:
                    logging.error(
                        'Cannot find the last definition node for accessed name %s given the scope stack %s. Added a node that accesses the name to local names within the current scope.',
                        name,
                        scope_stack)

        def scoped_node_visitor_callback_function(
                scope_stack: list[NodeProvidingScope],
                class_stack: list[ast.ClassDef],
                node: ast.AST
        ):
            # ast.Constant(value)
            if isinstance(node, ast.Constant):
                # Set the current type variable to be equivalent to `type(value)`
                set_node_to_be_instance_of(node, type(node.value))

            # ast.JoinedStr(values)
            if isinstance(node, ast.JoinedStr):
                # Set the current type variable to be equivalent to `str`
                set_node_to_be_instance_of(node, str)

            # ast.List(elts, ctx)
            if isinstance(node, ast.List):
                # Set the current type variable to be equivalent to `list`
                set_node_to_be_instance_of(node, list)

                for elt in node.elts:
                    if not isinstance(elt, ast.Starred):
                        # Set the type variable of `elt` as $ValueOf$ and $IterTargetOf$ the current type variable
                        non_equivalence_relation_graph.add_relation(node, elt, NonEquivalenceRelationType.ValueOf)
                        non_equivalence_relation_graph.add_relation(node, elt, NonEquivalenceRelationType.IterTargetOf)

                # Set $KeyOf$ the current type variable to be equivalent to `int`
                set_node_to_be_instance_of(
                    non_equivalence_relation_graph.get_or_create_related_node(node, NonEquivalenceRelationType.KeyOf),
                    int
                )

            # ast.Tuple(elts, ctx)
            if isinstance(node, ast.Tuple):
                # Set the current type variable to be equivalent to `tuple`
                set_node_to_be_instance_of(node, tuple)

                for i, elt in enumerate(node.elts):
                    if not isinstance(elt, ast.Starred):
                        # Set the type variable of `elt` as the $i$-th $ElementOf$ the current type variable
                        non_equivalence_relation_graph.add_relation(node, elt, NonEquivalenceRelationType.ElementOf, i)
                    else:
                        break

                # Set $KeyOf$ the current type variable to be equivalent to `int`
                set_node_to_be_instance_of(
                    non_equivalence_relation_graph.get_or_create_related_node(node, NonEquivalenceRelationType.KeyOf),
                    int
                )

            # ast.Set(elts)
            if isinstance(node, ast.Set):
                # Set the current type variable to be equivalent to `set`
                set_node_to_be_instance_of(node, set)

                for elt in node.elts:
                    if not isinstance(elt, ast.Starred):
                        # Set the type variable of `elt` as $IterTargetOf$ the current type variable
                        non_equivalence_relation_graph.add_relation(node, elt, NonEquivalenceRelationType.IterTargetOf)

            # ast.Dict(keys, values)
            if isinstance(node, ast.Dict):
                # Set the current type variable to be equivalent to `dict`
                set_node_to_be_instance_of(node, dict)

                for key_, value_ in zip(node.keys, node.values):
                    if key_ is not None:
                        # Set the type variable of `key` as $KeyOf$ and $IterTargetOf$ the current type variable
                        non_equivalence_relation_graph.add_relation(node, key_, NonEquivalenceRelationType.KeyOf)
                        non_equivalence_relation_graph.add_relation(node, key_, NonEquivalenceRelationType.IterTargetOf)
                        # Set the type variable of `value` as $ValueOf$ the current type variable
                        non_equivalence_relation_graph.add_relation(node, value_, NonEquivalenceRelationType.ValueOf)
                    else:
                        # as described in https://docs.python.org/3/reference/expressions.html#dictionary-displays
                        # Set the type variable of `value` to be equivalent to `collections.abc.Mapping`
                        set_node_to_be_instance_of(value_, collections.abc.Mapping)
                        # Set the $KeyOf$, $ValueOf$, and $IterTargetOf$ the type variable of `value` as equivalent to the the $KeyOf$, $ValueOf$, and $IterTargetOf$ the current type variable.
                        equivalence_relation_graph.set_equivalent(
                            non_equivalence_relation_graph.get_or_create_related_node(node,
                                                                                      NonEquivalenceRelationType.KeyOf),
                            non_equivalence_relation_graph.get_or_create_related_node(value_,
                                                                                      NonEquivalenceRelationType.KeyOf)
                        )

                        equivalence_relation_graph.set_equivalent(
                            non_equivalence_relation_graph.get_or_create_related_node(node,
                                                                                      NonEquivalenceRelationType.ValueOf),
                            non_equivalence_relation_graph.get_or_create_related_node(value_,
                                                                                      NonEquivalenceRelationType.ValueOf)
                        )

                        equivalence_relation_graph.set_equivalent(
                            non_equivalence_relation_graph.get_or_create_related_node(node,
                                                                                      NonEquivalenceRelationType.IterTargetOf),
                            non_equivalence_relation_graph.get_or_create_related_node(value_,
                                                                                      NonEquivalenceRelationType.IterTargetOf)
                        )

            # ast.Name(id, ctx)
            if isinstance(node, ast.Name):
                # Handle accessed name
                handle_node_that_accesses_name(
                    scope_stack,
                    node.id,
                    node,
                    isinstance(node.ctx, ast.Store)
                )

            # ast.Starred(value, ctx)
            if isinstance(node, ast.Starred):
                # Set the type variable of `value` to be equivalent to `collections.abc.Iterable`
                # according to https://docs.python.org/3/reference/expressions.html#grammar-token-python-grammar-starred_expression)
                set_node_to_be_instance_of(node.value, collections.abc.Iterable)

            # ast.UnaryOp(op, operand)
            if isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, ast.Not):
                    # Update the attribute counter of the type variable of `operand` with the attribute corresponding to `op`.
                    nodes_to_attribute_counters[node.operand][unaryop_to_attribute[type(node.op)]] += 1

                    # Set the current type variable as equivalent to the type variable of `operand`.
                    equivalence_relation_graph.set_equivalent(
                        node,
                        node.operand
                    )

            # ast.BinOp(left, op, right)
            if isinstance(node, ast.BinOp):
                # Update the attribute counter of the type variable of `left` with the attribute corresponding to `op`.
                nodes_to_attribute_counters[node.left][operator_to_attribute[type(node.op)]] += 1

                # Set the current type variable as equivalent to the type variable of `left` and `right`.
                equivalence_relation_graph.set_equivalent(
                    node,
                    node.left
                )

                equivalence_relation_graph.set_equivalent(
                    node,
                    node.right
                )

            # ast.BoolOp(op, values)
            if isinstance(node, ast.BoolOp):
                for value_ in node.values:
                    # Set the current type variable as equivalent to the type variable of `value`
                    equivalence_relation_graph.set_equivalent(
                        node,
                        value_
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
                        nodes_to_attribute_counters[left][cmpop_to_attribute[type(op)]] += 1

                        # Set the type variable of `left` as equivalent to the type variable of `right`.
                        equivalence_relation_graph.set_equivalent(
                            left,
                            right
                        )
                    elif isinstance(op, (ast.In, ast.NotIn)):
                        # based on https://docs.python.org/3/reference/expressions.html#membership-test-operations and https://discuss.python.org/t/deprecate-old-style-iteration-protocol/17863/7
                        # Update the attribute counter of the type variable of `right` with the attributes `__contains__` and `__iter__`.
                        nodes_to_attribute_counters[right]['__contains__'] += 1
                        nodes_to_attribute_counters[right]['__iter__'] += 1
                        # Set the type variable of `left` as $IterTargetOf$ the type variable of `right`.
                        non_equivalence_relation_graph.add_relation(right, left,
                                                                    NonEquivalenceRelationType.IterTargetOf)

                # Set the current type variable as equivalent to `bool`.
                set_node_to_be_instance_of(node, bool)

            # ast.Call(func, args, keywords, starargs, kwargs)
            if isinstance(node, ast.Call):
                # Update the attribute counter of the type variable of `func` with the attribute `__call__`.
                nodes_to_attribute_counters[node.func]['__call__'] += 1

                undetermined_number_of_parameters: bool = False

                for i, arg in enumerate(node.args):
                    if not isinstance(arg, ast.Starred):
                        # Set the type variable of `arg` as the $i$-th $ParameterOf$ the type variable of `func`.
                        non_equivalence_relation_graph.add_relation(node.func, arg,
                                                                    NonEquivalenceRelationType.ArgumentOf, i)
                    else:
                        undetermined_number_of_parameters = True
                        break

                for keyword in node.keywords:
                    if keyword.arg is None:
                        # Set the type variable of `keyword.value` as equivalent to `collections.abc.Mapping`.
                        # as described in https://docs.python.org/3/reference/expressions.html#dictionary-displays
                        set_node_to_be_instance_of(keyword.value, collections.abc.Mapping)

                        # Set the $KeyOf$ the type variable of `keyword.value` as equivalent to `str`.
                        set_node_to_be_instance_of(
                            non_equivalence_relation_graph.get_or_create_related_node(keyword.value,
                                                                                      NonEquivalenceRelationType.KeyOf),
                            str
                        )

                if node.keywords:
                    undetermined_number_of_parameters = True

                if undetermined_number_of_parameters:
                    # Create a dummy node to represent all parameters.
                    dummy_node_representing_all_parameters: ast.AST = add_new_dummy_node()
                    set_node_to_be_instance_of(dummy_node_representing_all_parameters, type(Ellipsis))

                    non_equivalence_relation_graph.add_relation(
                        node.func,
                        dummy_node_representing_all_parameters,
                        NonEquivalenceRelationType.ArgumentOf,
                        0
                    )


                # Set the current type variable as the $ReturnValueOf$ the type variable of `func`.
                non_equivalence_relation_graph.add_relation(node.func, node, NonEquivalenceRelationType.ReturnedValueOf)

            # ast.IfExp(test, body, orelse)
            if isinstance(node, ast.IfExp):
                # Set the current type variable as equivalent to the type variable of `body` and `orelse`.
                equivalence_relation_graph.set_equivalent(
                    node,
                    node.body
                )
                equivalence_relation_graph.set_equivalent(
                    node,
                    node.orelse
                )

            # ast.Attribute(value, attr, ctx)
            if isinstance(node, ast.Attribute):
                # Update the attribute counter of the type variable of `value` with `attr`.
                nodes_to_attribute_counters[node.value][node.attr] += 1

                # Set the current type variable as the $attr$-$AttrOf$ the type variable of `value`.
                non_equivalence_relation_graph.add_relation(node.value, node, NonEquivalenceRelationType.AttrOf,
                                                            node.attr)

            # ast.NamedExpr(target, value)
            if isinstance(node, ast.NamedExpr):
                # Set the current type variable as equivalent to the type variable of `target` and `value`.
                equivalence_relation_graph.set_equivalent(
                    node,
                    node.target
                )
                equivalence_relation_graph.set_equivalent(
                    node,
                    node.value
                )

            # ast.Subscript(value, slice, ctx)
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, (ast.Tuple, ast.Slice)):
                    # Set the current type variable as equivalent to the type variable of `value`.
                    equivalence_relation_graph.set_equivalent(
                        node,
                        node.value
                    )
                else:
                    # Set the current type variable as $ValueOf$ the type variable of `value`.
                    non_equivalence_relation_graph.add_relation(node.value, node, NonEquivalenceRelationType.ValueOf)

                    # Set the type variable of `slice` as $KeyOf$ the type variable of `value`.
                    non_equivalence_relation_graph.add_relation(node.value, node.slice,
                                                                NonEquivalenceRelationType.KeyOf)

                if isinstance(node.ctx, ast.Load):
                    # Update the attribute counter of the type variable of `value` with the attribute `__getitem__`.
                    nodes_to_attribute_counters[node.value]['__getitem__'] += 1

                if isinstance(node.ctx, ast.Store):
                    # Update the attribute counter of the type variable of `value` with the attribute `__setitem__`.
                    nodes_to_attribute_counters[node.value]['__setitem__'] += 1

            # ast.Slice(lower, upper, step)
            if isinstance(node, ast.Slice):
                # Set the current type variable as equivalent to `slice`.
                set_node_to_be_instance_of(node, slice)

                for value in (node.lower, node.upper, node.step):
                    if value is not None:
                        # Set the type variable of `value` as equivalent to `int`.
                        set_node_to_be_instance_of(value, int)

            # ast.ListComp(elt, generators)
            if isinstance(node, ast.ListComp):
                # Set the current type variable as equivalent to `list`.
                set_node_to_be_instance_of(node, list)

                # Set the type variable of `elt` as $ValueOf$ and $IterTargetOf$ the current type variable.
                non_equivalence_relation_graph.add_relation(node, node.elt, NonEquivalenceRelationType.ValueOf)
                non_equivalence_relation_graph.add_relation(node, node.elt, NonEquivalenceRelationType.IterTargetOf)

                # Set $KeyOf$ the current type variable as equivalent to `int`.
                set_node_to_be_instance_of(
                    non_equivalence_relation_graph.get_or_create_related_node(node, NonEquivalenceRelationType.KeyOf),
                    int
                )

            # ast.SetComp(elt, generators)
            if isinstance(node, ast.SetComp):
                # Set the current type variable as equivalent to `set`.
                set_node_to_be_instance_of(node, set)

                # Set the type variable of `elt` as $IterTargetOf$ the current type variable.
                non_equivalence_relation_graph.add_relation(node, node.elt, NonEquivalenceRelationType.IterTargetOf)

            # ast.GeneratorExp(elt, generators)
            if isinstance(node, ast.GeneratorExp):
                # Set the current type variable as equivalent to `collections.abc.Generator`.
                set_node_to_be_instance_of(node, collections.abc.Generator)

                # Set the type variable of `elt` as $IterTargetOf$ the current type variable.
                non_equivalence_relation_graph.add_relation(node, node.elt, NonEquivalenceRelationType.IterTargetOf)

            # ast.DictComp(key, value, generators)
            if isinstance(node, ast.DictComp):
                # Set the current type variable as equivalent to `dict`.
                set_node_to_be_instance_of(node, dict)

                # Set the type variable of `key` as $KeyOf$ and $IterTargetOf$ the current type variable.
                non_equivalence_relation_graph.add_relation(node, node.key, NonEquivalenceRelationType.KeyOf)
                non_equivalence_relation_graph.add_relation(node, node.key, NonEquivalenceRelationType.IterTargetOf)

                # Set the type variable of `value` as $ValueOf$ the current type variable.
                non_equivalence_relation_graph.add_relation(node, node.value, NonEquivalenceRelationType.ValueOf)

            # ast.comprehension(target, iter, ifs, is_async)
            if isinstance(node, ast.comprehension):
                if node.is_async:
                    # Update the attribute counter of the type variable of `iter` with the attribute `__aiter__`.
                    nodes_to_attribute_counters[node.iter]['__aiter__'] += 1
                else:
                    # Update the attribute counter of the type variable of `iter` with the attribute `__iter__`.
                    nodes_to_attribute_counters[node.iter]['__iter__'] += 1

                # Set the type variable of `target` as $IterTargetOf$ the type variable of `iter`.
                non_equivalence_relation_graph.add_relation(node.iter, node.target,
                                                            NonEquivalenceRelationType.IterTargetOf)

            # Statements

            # ast.Assign(targets, value, type_comment)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # Set the type variable of `target` as equivalent to the type variable of `value`.
                    equivalence_relation_graph.set_equivalent(
                        target,
                        node.value
                    )

            # ast.AnnAssign(target, annotation, value, simple)
            if isinstance(node, ast.AnnAssign):
                if node.value is not None:
                    # Set the type variable of `target` as equivalent to the type variable of `value`.
                    equivalence_relation_graph.set_equivalent(
                        node.target,
                        node.value
                    )

            # ast.AugAssign(target, op, value)
            if isinstance(node, ast.AugAssign):
                # Update the attribute counter of the type variable of `target` with the attribute corresponding to `op`.
                nodes_to_attribute_counters[node.target][operator_to_attribute[type(node.op)]] += 1

                # Set the type variable of `target` as equivalent to the type variable of `value`.
                equivalence_relation_graph.set_equivalent(
                    node.target,
                    node.value
                )

                if isinstance(node.target, ast.Name):
                    last_definition_node: typing.Optional[ast.AST] = get_last_definition_node(
                        scope_stack,
                        node.target.id,
                        False
                    )

                    if last_definition_node is not None:
                        # Set the last definition node to be equivalent to the current node
                        equivalence_relation_graph.set_equivalent(
                            node.target,
                            last_definition_node
                        )

            # ast.For(target, iter, body, orelse, type_comment)
            if isinstance(node, ast.For):
                # Update the attribute counter of the type variable of `iter` with the attribute `__iter__`.
                nodes_to_attribute_counters[node.iter]['__iter__'] += 1

                # Set the type variable of `target` as $IterTargetOf$ the type variable of `iter`.
                non_equivalence_relation_graph.add_relation(node.iter, node.target,
                                                            NonEquivalenceRelationType.IterTargetOf)

            # ast.AsyncFor(target, iter, body, orelse, type_comment)
            if isinstance(node, ast.AsyncFor):
                # Update the attribute counter of the type variable of `iter` with the attribute `__aiter__`.
                nodes_to_attribute_counters[node.iter]['__aiter__'] += 1

                # Set the type variable of `target` as $IterTargetOf$ the type variable of `iter`.
                non_equivalence_relation_graph.add_relation(node.iter, node.target,
                                                            NonEquivalenceRelationType.IterTargetOf)

            # ast.ExceptHandler(type, name, body)
            if isinstance(node, ast.ExceptHandler):
                if node.name is not None:
                    # Handle accessed name
                    handle_node_that_accesses_name(
                        scope_stack,
                        node.name,
                        node,
                        True
                    )

                    # Create `isinstance_node = ast.Name('isinstance', ast.Load())`, associate it with the built-in `isinstance` function.
                    isinstance_node = ast.Name('isinstance', ast.Load())
                    nodes_to_runtime_term_sets[isinstance_node].add(isinstance)

                    # Set the current type variable as the $0$-th $ParameterOf$ the type variable of `isinstance_node`.
                    non_equivalence_relation_graph.add_relation(isinstance_node, node,
                                                                NonEquivalenceRelationType.ArgumentOf, 0)

                    # Set the type variable of `type` as the $1$-st $ParameterOf$ the type variable of `isinstance_node`.
                    non_equivalence_relation_graph.add_relation(isinstance_node, node.type,
                                                                NonEquivalenceRelationType.ArgumentOf, 1)

            # ast.With(items, body, type_comment)
            if isinstance(node, ast.With):
                for withitem in node.items:
                    # Update the attribute counter of the type variable of `withitem.context_expr` with the attributes `__enter__`, `__exit__`.
                    nodes_to_attribute_counters[withitem.context_expr]['__enter__'] += 1
                    nodes_to_attribute_counters[withitem.context_expr]['__exit__'] += 1

                    if withitem.optional_vars is not None:
                        # `getattr_node = ast.Attribute(value=withitem.context_expr, attr='__enter__', ctx=ast.Load())`
                        getattr_node = ast.Attribute(value=withitem.context_expr, attr='__enter__', ctx=ast.Load())

                        # Set the type variable of `withitem.optional_vars` as the $ReturnValueOf$ the type variable of `getattr_node`.
                        non_equivalence_relation_graph.add_relation(getattr_node, withitem.optional_vars,
                                                                    NonEquivalenceRelationType.ReturnedValueOf)

            # ast.AsyncWith(items, body, type_comment)
            if isinstance(node, ast.AsyncWith):
                for withitem in node.items:
                    # Update the attribute counter of the type variable of `withitem.context_expr` with the attributes `__aenter__`, `__aexit__`.
                    nodes_to_attribute_counters[withitem.context_expr]['__aenter__'] += 1
                    nodes_to_attribute_counters[withitem.context_expr]['__aexit__'] += 1

                    if withitem.optional_vars is not None:
                        # `getattr_node = ast.Attribute(value=withitem.context_expr, attr='__aenter__', ctx=ast.Load())`
                        getattr_node = ast.Attribute(value=withitem.context_expr, attr='__aenter__', ctx=ast.Load())

                        # Set the type variable of `withitem.optional_vars` as the $ReturnValueOf$ the type variable of `getattr_node`.
                        non_equivalence_relation_graph.add_relation(getattr_node, withitem.optional_vars,
                                                                    NonEquivalenceRelationType.ReturnedValueOf)

            # ast.FunctionDef(name, args, body, decorator_list, returns, type_comment)
            # ast.AsyncFunctionDef(name, args, body, decorator_list, returns, type_comment)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Update the runtime term set of the current type variable.
                nodes_to_runtime_term_sets[node].add(node)

                # Handle accessed name.
                handle_node_that_accesses_name(
                    scope_stack,
                    node.name,
                    node,
                    True
                )

                # Initialize parameter list, return type set, yield value set, send value set of the current scope.
                nodes_providing_scope_to_parameter_lists[node] = []
                nodes_providing_scope_to_apparent_return_value_sets[node] = set()
                nodes_providing_scope_to_yield_value_sets[node] = set()
                nodes_providing_scope_to_send_value_sets[node] = set()

                if isinstance(node, ast.AsyncFunctionDef):
                    nodes_providing_scope_returning_coroutines.add(node)

            # ast.Lambda(args, body)
            if isinstance(node, ast.Lambda):
                # Update the runtime term set of the current type variable.
                nodes_to_runtime_term_sets[node].add(node)

                # Initialize parameter list, return type set, yield value set, send value set of the current scope.
                nodes_providing_scope_to_parameter_lists[node] = []
                nodes_providing_scope_to_apparent_return_value_sets[node] = set()
                nodes_providing_scope_to_yield_value_sets[node] = set()
                nodes_providing_scope_to_send_value_sets[node] = set()

                # Add the type variable of `body` to the return type set of the runtime term of the current type variable.
                nodes_providing_scope_to_apparent_return_value_sets[node].add(node.body)

            # ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults)
            if isinstance(node, ast.arguments):
                if scope_stack:
                    current_scope = scope_stack[-1]
                    for arg in node.posonlyargs + node.args:
                        # Handle accessed name
                        handle_node_that_accesses_name(
                            scope_stack,
                            arg.arg,
                            arg,
                            True
                        )

                        # Add the type variable of `arg` to the parameter type variable list of the current scope.
                        nodes_providing_scope_to_parameter_lists[current_scope].append(arg)
                else:
                    logging.error('Cannot handle ast.arguments outside of a scope!')

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

                    # Set the current type variable as the $YieldFromAwaitResultOf$ the type variable of `value`.
                    non_equivalence_relation_graph.add_relation(node.value, node,
                                                                NonEquivalenceRelationType.YieldFromAwaitResultOf)

                    # Update the attribute counter of the type variable of `value` with the attributes in `collections.abc.Iterator`.
                    set_node_to_be_instance_of(node.value, collections.abc.Iterator)

                    # Add the $IterTargetOf$ the type variable of `value` to the yield type set of the current scope.
                    nodes_providing_scope_to_yield_value_sets[current_scope].add(
                        non_equivalence_relation_graph.get_or_create_related_node(node.value,
                                                                                  NonEquivalenceRelationType.IterTargetOf)
                    )

                    # Add the $SendTargetOf$ the type variable of `value` to the send type set of the current scope.
                    nodes_providing_scope_to_yield_value_sets[current_scope].add(
                        non_equivalence_relation_graph.get_or_create_related_node(node.value,
                                                                                  NonEquivalenceRelationType.SendTargetOf)
                    )
                else:
                    logging.error('Cannot handle ast.YieldFrom outside of a scope!')

            # ast.ClassDef(name, bases, keywords, starargs, kwargs, body, decorator_list)
            if isinstance(node, ast.ClassDef):
                # Handle accessed name.
                handle_node_that_accesses_name(
                    scope_stack,
                    node.name,
                    node,
                    True
                )

                # Update the runtime term set of the current type variable.
                if node in top_level_class_definitions_to_runtime_classes:
                    runtime_class = top_level_class_definitions_to_runtime_classes[node]
                    nodes_to_runtime_term_sets[node].add(runtime_class)

            # ast.Await(value)
            if isinstance(node, ast.Await):
                # Update the attribute counter of the type variable of `value` with the attribute `__await__`.
                nodes_to_attribute_counters[node.value]['__await__'] += 1

                # Set the current type variable as the $YieldFromAwaitResultOf$ of the type variable of `value`
                non_equivalence_relation_graph.add_relation(node.value, node,
                                                            NonEquivalenceRelationType.YieldFromAwaitResultOf)

                # Set the type variable of `value` as equivalent to `collections.abc.Awaitable`.
                set_node_to_be_instance_of(node.value, collections.abc.Awaitable)

            if isinstance(node, ast.Global):
                for name in node.names:
                    # Handle global name declaration
                    handle_explicit_global_name_declaration(scope_stack, name)

            if isinstance(node, ast.Nonlocal):
                for name in node.names:
                    # Handle nonlocal name declaration
                    handle_explicit_nonlocal_name_declaration(scope_stack, name)

        return scoped_node_visitor_callback_function

    for module_name, module_node in module_names_to_module_nodes.items():
        ScopedNodeVisitor(get_scoped_node_visitor_callback_function(module_name)).visit(module_node)

    # Refine equivalence relations.

    # All values returned, yielded, and sent from a node providing scope are equivalent.

    for node_providing_scope, return_value_set in nodes_providing_scope_to_apparent_return_value_sets.items():
        for first, second in itertools.pairwise(return_value_set):
            equivalence_relation_graph.set_equivalent(first, second)

    for node_providing_scope, yield_value_set in nodes_providing_scope_to_yield_value_sets.items():
        for first, second in itertools.pairwise(yield_value_set):
            equivalence_relation_graph.set_equivalent(first, second)

    for node_providing_scope, send_value_set in nodes_providing_scope_to_send_value_sets.items():
        for first, second in itertools.pairwise(send_value_set):
            equivalence_relation_graph.set_equivalent(first, second)

    # The first parameter (`self`) of all instance methods within a runtime class are equivalent.
    # The first parameter (`cls`) of all classmethods within a runtime class are equivalent with the class definition.
    # Also update their attribute counters and runtime sets.
    for top_level_class_definition, runtime_class in top_level_class_definitions_to_runtime_classes.items():
        first_parameter_of_instance_methods = []
        first_parameter_of_classmethods = []

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
                    posargs,
                    vararg,
                    kwonlyargs,
                    kwarg
                ) = get_parameters(function_definition)
                if posargs:
                    first_parameter: ast.arg = posargs[0]

                    if is_classmethod:
                        first_parameter_of_classmethods.append(first_parameter)
                    if not is_staticmethod and not is_classmethod:
                        first_parameter_of_instance_methods.append(first_parameter)

        for first_parameter_of_instance_method in first_parameter_of_instance_methods:
            set_node_to_be_instance_of(first_parameter_of_instance_method, runtime_class)

        for first, second in itertools.pairwise(first_parameter_of_instance_methods):
            equivalence_relation_graph.set_equivalent(first, second)

        for first_parameter_of_classmethod in first_parameter_of_classmethods:
            set_node_to_be_runtime_class(first_parameter_of_classmethod, runtime_class)
            equivalence_relation_graph.set_equivalent(top_level_class_definition, first_parameter_of_classmethod)

    # Resolve the parameter lists and (real) return value sets of nodes providing scope.

    nodes_providing_scope_to_parameter_lists_and_return_value_sets: defaultdict[
        NodeProvidingScope,
        tuple[list[ast.arg], set[ast.AST]]
    ] = defaultdict(lambda: (list(), set()))

    nodes_providing_scope_set = set().union(
        nodes_providing_scope_to_parameter_lists.keys(),
        nodes_providing_scope_to_apparent_return_value_sets.keys(),
        nodes_providing_scope_to_yield_value_sets.keys(),
        nodes_providing_scope_to_send_value_sets.keys()
    )

    for node_providing_scope in nodes_providing_scope_set:
        parameter_list = nodes_providing_scope_to_parameter_lists[node_providing_scope]
        apparent_return_value_set = nodes_providing_scope_to_apparent_return_value_sets[node_providing_scope]
        yield_value_set = nodes_providing_scope_to_yield_value_sets[node_providing_scope]
        send_value_set = nodes_providing_scope_to_send_value_sets[node_providing_scope]

        # If there is no apparent return value, then add a dummy node to represent the return value of None.
        if not apparent_return_value_set:
            augmented_apparent_return_value_set: set[ast.AST] = set()

            return_value: ast.AST = add_new_dummy_node()
            set_node_to_be_instance_of(return_value, type(None))
            augmented_apparent_return_value_set.add(return_value)
        else:
            augmented_apparent_return_value_set: set[ast.AST] = apparent_return_value_set.copy()

        # non-async functions returning generators
        if (
                node_providing_scope in nodes_providing_scope_returning_generators
                and not node_providing_scope in nodes_providing_scope_returning_coroutines
        ):
            generator_return_value: ast.AST = add_new_dummy_node()

            set_node_to_be_instance_of(generator_return_value, collections.abc.Generator)

            for yield_value in yield_value_set:
                non_equivalence_relation_graph.add_relation(generator_return_value, yield_value,
                                                            NonEquivalenceRelationType.IterTargetOf)

            for send_value in send_value_set:
                non_equivalence_relation_graph.add_relation(generator_return_value, send_value,
                                                            NonEquivalenceRelationType.SendTargetOf)

            for apparent_return_value in augmented_apparent_return_value_set:
                non_equivalence_relation_graph.add_relation(generator_return_value, apparent_return_value,
                                                            NonEquivalenceRelationType.YieldFromAwaitResultOf)

            return_value_set = {generator_return_value}
        # async functions returning generators
        elif (
                node_providing_scope in nodes_providing_scope_returning_generators
                and node_providing_scope in nodes_providing_scope_returning_coroutines
        ):
            async_generator_return_value: ast.AST = add_new_dummy_node()

            set_node_to_be_instance_of(async_generator_return_value, collections.abc.AsyncGenerator)

            for yield_value in yield_value_set:
                non_equivalence_relation_graph.add_relation(async_generator_return_value, yield_value,
                                                            NonEquivalenceRelationType.IterTargetOf)

            for send_value in send_value_set:
                non_equivalence_relation_graph.add_relation(async_generator_return_value, send_value,
                                                            NonEquivalenceRelationType.SendTargetOf)

            return_value_set = {async_generator_return_value}
        # async functions not returning generators
        elif (
                not node_providing_scope in nodes_providing_scope_returning_generators
                and node_providing_scope in nodes_providing_scope_returning_coroutines
        ):
            coroutine_return_value: ast.AST = add_new_dummy_node()

            set_node_to_be_instance_of(coroutine_return_value, collections.abc.Coroutine)

            for yield_value in yield_value_set:
                non_equivalence_relation_graph.add_relation(coroutine_return_value, yield_value,
                                                            NonEquivalenceRelationType.IterTargetOf)

            for send_value in send_value_set:
                non_equivalence_relation_graph.add_relation(coroutine_return_value, send_value,
                                                            NonEquivalenceRelationType.SendTargetOf)

            for apparent_return_value in augmented_apparent_return_value_set:
                non_equivalence_relation_graph.add_relation(coroutine_return_value, apparent_return_value,
                                                            NonEquivalenceRelationType.YieldFromAwaitResultOf)

            return_value_set = {coroutine_return_value}
        # non-async functions not returning generators
        else:
            return_value_set = augmented_apparent_return_value_set

        nodes_providing_scope_to_parameter_lists_and_return_value_sets[node_providing_scope] = (
            parameter_list,
            return_value_set
        )

    # Add attribute counters and runtime term sets for nodes providing scope.
    for node_providing_scope, (parameter_list, return_value_set) in nodes_providing_scope_to_parameter_lists_and_return_value_sets.items():
        if isinstance(node_providing_scope, FunctionDefinition):
            set_node_to_be_instance_of(node_providing_scope, collections.abc.Callable)

            (
                posargs,
                vararg,
                kwonlyargs,
                kwarg
            ) = get_parameters(node_providing_scope)

            if vararg is not None or kwonlyargs or kwarg is not None:
                # Create a dummy node to represent all parameters.
                dummy_node_representing_all_parameters: ast.AST = add_new_dummy_node()
                set_node_to_be_instance_of(dummy_node_representing_all_parameters, type(Ellipsis))

                non_equivalence_relation_graph.add_relation(
                    node_providing_scope,
                    dummy_node_representing_all_parameters,
                    NonEquivalenceRelationType.ArgumentOf,
                    0
                )
            else:
                for i, parameter in enumerate(parameter_list):
                    non_equivalence_relation_graph.add_relation(
                        node_providing_scope,
                        parameter,
                        NonEquivalenceRelationType.ArgumentOf,
                        i
                    )

            for return_value in return_value_set:
                non_equivalence_relation_graph.add_relation(
                    node_providing_scope,
                    return_value,
                    NonEquivalenceRelationType.ReturnedValueOf
                )

    return (
        nodes_to_attribute_counters,
        nodes_to_runtime_term_sets,
        nodes_providing_scope_to_parameter_lists_and_return_value_sets,
        equivalence_relation_graph,
        non_equivalence_relation_graph
    )
