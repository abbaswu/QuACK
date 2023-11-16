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
    assert str(parse('Tuple[<partial None>, <partial None>]')) == 'builtins.tuple[None, None]'
    assert str(parse('Tuple[Union[Any, builtins.str], Union[Any, builtins.str]]')) == 'builtins.tuple[typing.Union[typing.Any, builtins.str], typing.Union[typing.Any, builtins.str]]'
    assert str(parse('Tuple[<partial list[?]>, builtins.list[builtins.tuple[builtins.str]]]')) == 'builtins.tuple[builtins.list[typing.Any], builtins.list[builtins.tuple[builtins.str, ...]]]'
    assert str(parse('def (max_workers: Union[None, builtins.int] =, thread_name_prefix: builtins.str =, initializer: Union[None, def (*Any, **Any)] =, initargs: builtins.tuple[Any] =) -> concurrent.futures.thread.ThreadPoolExecutor')) == 'typing.Callable[[typing.Union[None, builtins.int], builtins.str, typing.Union[None, typing.Callable[..., None]], builtins.tuple[typing.Any, ...]], concurrent.futures.thread.ThreadPoolExecutor]'
    assert str(parse('Tuple[]')) == 'builtins.tuple'
    assert str(parse("Literal['+a']")) == "typing.Literal['+a']"
    assert str(parse('Overload(def (message: builtins.object, errors: Tuple[] =, details: builtins.object =, response: builtins.object =, error_info: builtins.object =) -> google.api_core.exceptions.GoogleAPICallError, def (message: builtins.object, errors: builtins.object=, details: builtins.object =, response: builtins.object =, error_info: builtins.object =) -> google.api_core.exceptions.GoogleAPICallError, def (message: builtins.object, errors: builtins.object =, details: Tuple[] =, response: builtins.object =,error_info: builtins.object =) -> google.api_core.exceptions.GoogleAPICallError, def (message: builtins.object, errors: Tuple[] =, details: Tuple[] =, response: builtins.object =, error_info: builtins.object =) -> google.api_core.exceptions.GoogleAPICallError)')) == 'typing.Callable[[builtins.object, typing.Union[builtins.object, builtins.tuple], typing.Union[builtins.object, builtins.tuple], builtins.object, builtins.object], google.api_core.exceptions.GoogleAPICallError]'
