import ast
import typing


# Node in Python providing a function-like scope
NodeProvidingScope = typing.Union[
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp
]


class ScopedNodeVisitor(ast.NodeVisitor):
    def __init__(self, callback: typing.Callable[[list[NodeProvidingScope], list[ast.ClassDef], ast.AST], None]):
        self.scope_stack: list[NodeProvidingScope] = []
        self.class_stack: list[ast.ClassDef] = []
        self.callback = callback

    # Before being overwritten, this method calls self.visit on each child node.
    # We need to do this
    # Either by manually calling self.visit on each child node (if we want to customize the visit order)
    # Or by calling super().generic_visit(node)
    def generic_visit(self, node):
        self.callback(self.scope_stack, self.class_stack, node)

        if isinstance(node, NodeProvidingScope):
            self.scope_stack.append(node)

            # ListComp(expr elt, comprehension * generators)
            # SetComp(expr elt, comprehension * generators)
            # GeneratorExp(expr elt, comprehension * generators)
            # Visit generators before visiting elt
            if isinstance(node, (
                ast.ListComp,
                ast.SetComp,
                ast.GeneratorExp
            )):
                for generator in node.generators:
                    self.visit(generator)
                self.visit(node.elt)
            # DictComp(expr key, expr value, comprehension * generators)
            # Visit generators before visiting key, value
            elif isinstance(node, ast.DictComp):
                for generator in node.generators:
                    self.visit(generator)
                self.visit(node.key)
                self.visit(node.value)
            else:
                super().generic_visit(node)

            self.scope_stack.pop()
        elif isinstance(node, ast.ClassDef):
            self.class_stack.append(node)

            super().generic_visit(node)

            self.class_stack.pop()
        # Visit assignments from value to targets
        # Assign(expr* targets, expr value, string? type_comment)
        elif isinstance(node, ast.Assign):
            self.visit(node.value)
            for target in node.targets:
                self.visit(target)
        # AugAssign(expr target, operator op, expr value)
        elif isinstance(node, ast.AugAssign):
            self.visit(node.value)
            self.visit(node.op)
            self.visit(node.target)
        # AnnAssign(expr target, expr annotation, expr? value, int simple)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self.visit(node.value)
            self.visit(node.annotation)
            self.visit(node.target)
        # NamedExpr(expr target, expr value)
        elif isinstance(node, ast.NamedExpr):
            self.visit(node.value)
            self.visit(node.target)
        else:
            super().generic_visit(node)
