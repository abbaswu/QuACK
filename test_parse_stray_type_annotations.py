"""
To generate an HTML coverage report:

- Run Coverage: coverage run --source=parse_stray_type_annotations test_parse_stray_type_annotations.py
- Generate an HTML report: coverage html

References:

- https://chat.openai.com/share/0977e84d-91b6-4e2b-addc-0cc53a0ff5da
"""

from type_inference_result import TypeInferenceResult
from parse_stray_type_annotations import parser, handle_type_annotation_tree


def parse(
    type_annotation_string: str
) -> TypeInferenceResult:
    type_annotation_tree: Tree = parser.parse(type_annotation_string)
    return handle_type_annotation_tree(
        type_annotation_tree,
        dict()
    )

if __name__ == '__main__':
    assert str(parse('None')) == 'None'
    assert str(parse('builtins.int')) == 'builtins.int'
    assert str(parse('Tuple[Any, def (*args: Any, **kw: Any) -> ctypes.Structure]')) == 'builtins.tuple[typing.Any, typing.Callable[..., ctypes.Structure]]'
    assert str(parse('Any')) == 'typing.Any'
    assert str(parse('Tuple[builtins.int, builtins.int, builtins.int]')) == 'builtins.tuple[builtins.int, builtins.int, builtins.int]'
    assert str(parse('builtins.tuple[builtins.int]')) == 'builtins.tuple[builtins.int, ...]'
    assert str(parse('<nothing>')) == 'typing.NoReturn'
