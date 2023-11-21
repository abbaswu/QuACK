"""
NOTE: This module is intended to be used as a singleton.
Initialize it by calling get_parameter_lists_and_symbolic_return_values() once.
"""

import _ast
import ast
import typing
from collections import defaultdict

from get_parameters import get_parameters
from node_visitor import NodeProvidingScope, ScopedNodeVisitor


nodes_to_parameter_lists_and_symbolic_return_values: defaultdict[
    _ast.AST,
    tuple[list[ast.arg], _ast.AST]
] = defaultdict(lambda: ([], _ast.AST()))


def get_parameter_lists_and_symbolic_return_values(
    module_nodes: typing.Iterable[ast.Module]
):
    def collect_parameter_return_value_information_callback(
            scope_stack: list[NodeProvidingScope],
            class_stack: list[ast.ClassDef],
            node: _ast.AST
    ):
        # ast.FunctionDef(name, args, body, decorator_list, returns, type_comment)
        # ast.AsyncFunctionDef(name, args, body, decorator_list, returns, type_comment)
        # ast.Lambda(args, body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Initialize parameter list.
            (
                posargs,
                vararg,
                kwonlyargs,
                kwarg
            ) = get_parameters(node)

            nodes_to_parameter_lists_and_symbolic_return_values[node] = (
                posargs,
                _ast.AST()
            )

    for module_node in module_nodes:
        ScopedNodeVisitor(collect_parameter_return_value_information_callback).visit(module_node)
