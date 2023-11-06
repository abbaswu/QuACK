import ast

from scoped_node_visitor import ScopedNodeVisitor, NodeProvidingScope

if __name__ == '__main__':
    import shell_sort.shell_sort

    with open(shell_sort.shell_sort.__file__, 'r') as fp:
        node = ast.parse(fp.read())

    def callback(scope: list[NodeProvidingScope], classes: list[ast.ClassDef], node: ast.AST):
        print(scope, classes, ast.unparse(node))

    ScopedNodeVisitor(callback).visit(node)
