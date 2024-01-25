import _ast
import ast
import json
import typing


node_to_textual_representation_dict: dict[_ast.AST, str] = dict()


def initialize(
    module_name_to_module_node_dict: typing.Mapping[str, ast.Module]
):
    for module_name, module_node in module_name_to_module_node_dict.items():
        for node in ast.walk(module_node):
            node_to_textual_representation_dict[node] = get_textual_representation(
                node,
                module_name,
                getattr(node, 'lineno', None),
                ast.unparse(node)
            )


def get_textual_representation(
        node: _ast.AST,
        module_name: typing.Optional[str] = None,
        lineno: typing.Optional[int] = None,
        textual_description: typing.Optional[str] = None
):
    textual_representation_components: list[str] = [str(node)]

    if module_name is not None:
        if lineno is not None:
            textual_representation_components.append(module_name + str(lineno))

    if textual_description is not None:
        textual_representation_components.append(json.dumps(textual_description))

    return ' '.join(textual_representation_components)
