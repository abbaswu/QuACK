import _ast
import ast


def node_to_string(node: _ast.AST) -> str:
    return f'{}:{}:{ast.unparse(node)}'