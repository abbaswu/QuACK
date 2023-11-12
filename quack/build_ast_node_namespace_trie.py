"""
Nodes that access names:

- ast.ClassDef
- ast.FunctionDef, ast.AsyncFunctionDef
- ast.arg
- ast.ExceptHandler
- ast.Name
"""
import _ast
import ast
import typing

from node_visitor import get_child_nodes
from parameter_lists_and_symbolic_return_values import nodes_to_parameter_lists_and_symbolic_return_values
from trie import TrieNode


def add_node_that_accesses_name(
        namespace_defining_trie_node: TrieNode[str, dict[str, set[ast.AST]]],
        node: ast.AST,
        name: str
):
    if namespace_defining_trie_node.value is None:
        namespace_defining_trie_node.value = dict()

    if name not in namespace_defining_trie_node.value:
        namespace_defining_trie_node.value[name] = set()

    namespace_defining_trie_node.value[name].add(node)


class ModuleLevelASTNodeNamespaceTrieBuilder:
    def __init__(self, root: TrieNode[str, dict[str, set[ast.AST]]], module_name: str):
        self.namespace_defining_trie_node_stack: list[TrieNode[str, dict[str, set[ast.AST]]]] = [root]
        self.module_name = module_name

    def visit(self, node: _ast.AST):
        current_namespace_defining_trie_node = self.namespace_defining_trie_node_stack[-1]

        # ast.Module(body, type_ignores)
        if isinstance(node, ast.Module):
            add_node_that_accesses_name(
                current_namespace_defining_trie_node,
                node,
                self.module_name,
            )

            new_namespace_defining_trie_node: TrieNode[str, dict[str, set[ast.AST]]] = TrieNode()
            current_namespace_defining_trie_node.children[self.module_name] = new_namespace_defining_trie_node
            self.namespace_defining_trie_node_stack.append(new_namespace_defining_trie_node)

            for child_node in get_child_nodes(node):
                self.visit(child_node)

            self.namespace_defining_trie_node_stack.pop()
        # ast.ClassDef(name, bases, keywords, body, decorator_list)
        elif isinstance(node, ast.ClassDef):
            add_node_that_accesses_name(
                current_namespace_defining_trie_node,
                node,
                node.name,
            )

            new_namespace_defining_trie_node: TrieNode[str, dict[str, set[ast.AST]]] = TrieNode()
            current_namespace_defining_trie_node.children[node.name] = new_namespace_defining_trie_node
            self.namespace_defining_trie_node_stack.append(new_namespace_defining_trie_node)

            for child_node in get_child_nodes(node):
                self.visit(child_node)

            self.namespace_defining_trie_node_stack.pop()
        # ast.FunctionDef(name, args, body, decorator_list, returns, type_comment)
        # ast.AsyncFunctionDef(name, args, body, decorator_list, returns, type_comment)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_node_that_accesses_name(
                current_namespace_defining_trie_node,
                node,
                node.name,
            )

            new_namespace_defining_trie_node: TrieNode[str, dict[str, set[ast.AST]]] = TrieNode()
            current_namespace_defining_trie_node.children[node.name] = new_namespace_defining_trie_node
            self.namespace_defining_trie_node_stack.append(new_namespace_defining_trie_node)

            # Add parameters and return value to namespace.
            if node in nodes_to_parameter_lists_and_symbolic_return_values:
                parameter_list, symbolic_return_value = nodes_to_parameter_lists_and_symbolic_return_values[node]
                for parameter in parameter_list:
                    add_node_that_accesses_name(
                        new_namespace_defining_trie_node,
                        parameter,
                        parameter.arg,
                    )
                add_node_that_accesses_name(
                    new_namespace_defining_trie_node,
                    symbolic_return_value,
                    'return',
                )

            for child_node in get_child_nodes(node):
                self.visit(child_node)

            self.namespace_defining_trie_node_stack.pop()
        else:
            if isinstance(node, ast.ExceptHandler):
                if node.name is not None:
                    add_node_that_accesses_name(
                        current_namespace_defining_trie_node,
                        node,
                        node.name,
                    )
            elif isinstance(node, ast.Name):
                add_node_that_accesses_name(
                    current_namespace_defining_trie_node,
                    node,
                    node.id,
                )

            for child_node in get_child_nodes(node):
                self.visit(child_node)


def build_ast_node_namespace_trie(
        module_names: typing.Iterable[str],
        module_nodes: typing.Iterable[ast.Module]
) -> TrieNode[str, dict[str, set[ast.AST]]]:
    root: TrieNode[str, dict[str, set[ast.AST]]] = TrieNode()

    for module_name, module_node in zip(module_names, module_nodes):
        ModuleLevelASTNodeNamespaceTrieBuilder(root, module_name).visit(module_node)

    return root


def get_node_to_name_component_tuple_dict(
        ast_node_namespace_trie: TrieNode[str, dict[str, set[ast.AST]]]
) -> dict[ast.AST, tuple[str, ...]]:
    def recursive_function(current_namespace_root: TrieNode[str, dict[str, set[ast.AST]]]) -> typing.Generator[
        tuple[ast.AST, tuple[str, ...]], None, None
    ]:
        if current_namespace_root.value is not None:
            for name, node_set in current_namespace_root.value.items():
                for node in node_set:
                    yield node, (name,)
        for child_namespace_name, child_namespace_root in current_namespace_root.children.items():
            for node, relative_name_component_tuple in recursive_function(child_namespace_root):
                yield node, (child_namespace_name, ) + relative_name_component_tuple

    return dict(recursive_function(ast_node_namespace_trie))
