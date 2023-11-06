"""
Nodes that access names:

- ast.ClassDef
- ast.FunctionDef, ast.AsyncFunctionDef
- ast.arg
- ast.ExceptHandler
- ast.Name
"""
import ast
import typing

from trie import TrieNode


class ModuleLevelASTNodeNamespaceTrieBuilder(ast.NodeVisitor):
    def __init__(self, root: TrieNode[str, dict[str, set[ast.AST]]], module_name: str):
        self.namespace_defining_trie_node_stack: list[TrieNode[str, dict[str, set[ast.AST]]]] = [root]
        self.module_name = module_name

    def handle_node_that_accesses_name(self, node: ast.AST, name: str, defines_namespace: bool = False):
        current_namespace_defining_trie_node = self.namespace_defining_trie_node_stack[-1]

        if current_namespace_defining_trie_node.value is None:
            current_namespace_defining_trie_node.value = dict()

        if name not in current_namespace_defining_trie_node.value:
            current_namespace_defining_trie_node.value[name] = set()

        current_namespace_defining_trie_node.value[name].add(node)

        if defines_namespace:
            namespace_root: TrieNode[str, dict[str, set[ast.AST]]] = TrieNode()
            current_namespace_defining_trie_node.children[name] = namespace_root
            self.namespace_defining_trie_node_stack.append(namespace_root)

            self.generic_visit(node)

            self.namespace_defining_trie_node_stack.pop()
        else:
            self.generic_visit(node)

    def visit_Module(self, node: ast.Module):
        self.handle_node_that_accesses_name(
            node,
            self.module_name,
            True
        )

    def visit_ClassDef(self, node: ast.ClassDef):
        self.handle_node_that_accesses_name(
            node,
            node.name,
            True
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.handle_node_that_accesses_name(
            node,
            node.name,
            True
        )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.handle_node_that_accesses_name(
            node,
            node.name,
            True
        )

    def visit_arg(self, node: ast.arg):
        self.handle_node_that_accesses_name(
            node,
            node.arg,
            False
        )

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name is not None:
            self.handle_node_that_accesses_name(
                node,
                node.name,
                False
            )

    def visit_Name(self, node: ast.Name):
        self.handle_node_that_accesses_name(
            node,
            node.id,
            False
        )


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
