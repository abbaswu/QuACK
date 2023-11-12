import _ast
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


def get_child_nodes(node: _ast.AST) -> typing.Iterator[_ast.AST]:
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, _ast.AST):
                    yield item
        elif isinstance(value, _ast.AST):
            yield value


class AsyncScopedNodeVisitor:
    def __init__(self, async_callback: typing.Callable[[list[NodeProvidingScope], list[ast.ClassDef], ast.AST], typing.Awaitable]):
        self.scope_stack: list[NodeProvidingScope] = []
        self.class_stack: list[ast.ClassDef] = []
        self.async_callback = async_callback

    async def visit(self, node: _ast.AST):
        # Run the callback on each encountered node.
        await self.async_callback(self.scope_stack, self.class_stack, node)

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
                    await self.visit(generator)
                await self.visit(node.elt)
            # DictComp(expr key, expr value, comprehension * generators)
            # Visit generators before visiting key, value
            elif isinstance(node, ast.DictComp):
                for generator in node.generators:
                    await self.visit(generator)
                await self.visit(node.key)
                await self.visit(node.value)
            # Visit child nodes in default order
            else:
                for child_node in get_child_nodes(node):
                    await self.visit(child_node)

            self.scope_stack.pop()
        elif isinstance(node, ast.ClassDef):
            self.class_stack.append(node)

            # Visit child nodes in default order
            for child_node in get_child_nodes(node):
                await self.visit(child_node)

            self.class_stack.pop()
        # Visit assignments from value to targets
        # Assign(expr* targets, expr value, string? type_comment)
        elif isinstance(node, ast.Assign):
            await self.visit(node.value)
            for target in node.targets:
                await self.visit(target)
        # AugAssign(expr target, operator op, expr value)
        elif isinstance(node, ast.AugAssign):
            await self.visit(node.value)
            await self.visit(node.op)
            await self.visit(node.target)
        # AnnAssign(expr target, expr annotation, expr? value, int simple)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                await self.visit(node.value)
            await self.visit(node.annotation)
            await self.visit(node.target)
        # NamedExpr(expr target, expr value)
        elif isinstance(node, ast.NamedExpr):
            await self.visit(node.value)
            await self.visit(node.target)
        else:
            # Visit child nodes in default order
            for child_node in get_child_nodes(node):
                await self.visit(child_node)


class ScopedNodeVisitor:
    def __init__(self, callback: typing.Callable[[list[NodeProvidingScope], list[ast.ClassDef], ast.AST], None]):
        self.scope_stack: list[NodeProvidingScope] = []
        self.class_stack: list[ast.ClassDef] = []
        self.callback = callback

    def visit(self, node):
        # Run the callback on each encountered node.
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
            # Visit child nodes in default order
            else:
                for child_node in get_child_nodes(node):
                    self.visit(child_node)

            self.scope_stack.pop()
        elif isinstance(node, ast.ClassDef):
            self.class_stack.append(node)

            # Visit child nodes in default order
            for child_node in get_child_nodes(node):
                self.visit(child_node)

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
            # Visit child nodes in default order
            for child_node in get_child_nodes(node):
                self.visit(child_node)


class AsyncEvaluationOrderBottomUpNodeVisitor:
    def __init__(self, async_callback: typing.Callable[[ast.AST], typing.Awaitable]):
        self.async_callback = async_callback

    async def visit(self, node: _ast.AST):
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
                await self.visit(generator)
            await self.visit(node.elt)
        # DictComp(expr key, expr value, comprehension * generators)
        # Visit generators before visiting key, value
        elif isinstance(node, ast.DictComp):
            for generator in node.generators:
                await self.visit(generator)
            await self.visit(node.key)
            await self.visit(node.value)
        # Visit assignments from value to targets
        # Assign(expr* targets, expr value, string? type_comment)
        elif isinstance(node, ast.Assign):
            await self.visit(node.value)
            for target in node.targets:
                await self.visit(target)
        # AugAssign(expr target, operator op, expr value)
        elif isinstance(node, ast.AugAssign):
            await self.visit(node.value)
            await self.visit(node.op)
            await self.visit(node.target)
        # AnnAssign(expr target, expr annotation, expr? value, int simple)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                await self.visit(node.value)
            await self.visit(node.annotation)
            await self.visit(node.target)
        # NamedExpr(expr target, expr value)
        elif isinstance(node, ast.NamedExpr):
            await self.visit(node.value)
            await self.visit(node.target)
        # Visit calls from args to keywords to func.
        # Call(expr func, expr* args, keyword* keywords)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                await self.visit(arg)
            for keyword in node.keywords:
                await self.visit(keyword)
            await self.visit(node.func)
        else:
            # Visit child nodes in default order
            for child_node in get_child_nodes(node):
                await self.visit(child_node)

        # Run the callback on each encountered node.
        await self.async_callback(node)
