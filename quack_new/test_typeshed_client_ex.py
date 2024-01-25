"""
To generate an HTML coverage report for typeshed_client_ex under test:

- Run Coverage: coverage run --source=typeshed_client_ex test_typeshed_client_ex.py
- Generate an HTML report: coverage html

For writing tests for the OtherRelationsGraph class, we need to cover the following:

    Basic graph operations: initialization, adding relations, and copying the graph.
    Retrieval operations: get or create related nodes and get in/out edges based on relation tuples/types.
    Edge cases: handling nodes that don't exist, using parameters or not, etc.

Here's a basic test suite:
"""

from typeshed_client_ex.client import Client
from typeshed_client_ex.runtime_class_name_tuples_and_typeshed_class_name_tuples import \
    runtime_class_name_tuple_to_typeshed_class_name_tuple, typeshed_class_name_tuple_to_runtime_class
from typeshed_client_ex.type_definitions import TypeshedClass, TypeshedFunction, TypeshedTypeVariable, Union, \
    Subscription, instantiate_type_variables_in_class_definition, \
    get_comprehensive_type_annotations_for_parameters_and_return_values, to_runtime_class, from_runtime_class

client = Client()

assert isinstance(client.look_up_name('builtins', '_T_co'), TypeshedTypeVariable)

assert client.look_up_name('builtins', 'memoryview') == TypeshedClass('builtins', 'memoryview')

assert client.look_up_name('builtins', 'Sequence') == TypeshedClass('typing', 'Sequence')

assert client.look_up_name('collections.abc', 'Generator') == TypeshedClass('typing', 'Generator')

assert client.look_up_name('builtins', 'Sequence') == TypeshedClass('typing', 'Sequence')

assert client.look_up_name('builtins', 'NoneType') == TypeshedClass('types', 'NoneType')

assert client.look_up_name('builtins', 'function') == TypeshedClass('types', 'FunctionType')

assert client.look_up_name('builtins', '_LiteralInteger') == TypeshedClass('builtins', 'int')

assert client.look_up_name('builtins', 'divmod') == TypeshedFunction('builtins', 'divmod')

anystr = Union(
    frozenset(
        [
            TypeshedClass('builtins', 'bytes'),
            TypeshedClass('builtins', 'str')
        ]
    )
)

assert client.look_up_name('typing', 'AnyStr') == anystr

re_compile_lookup_result = client.look_up_name('re', 'compile')
assert re_compile_lookup_result == TypeshedFunction('re', 'compile')
function_definitions = client.get_function_definition(re_compile_lookup_result)
re_compile_return_value_type_annotations = {
    function_definition.return_value_type_annotation
    for function_definition in function_definitions
}
assert re_compile_return_value_type_annotations == {
    Subscription(
        TypeshedClass('re', 'Pattern'),
        (anystr,)
    )
}

re_pattern_lookup_result = client.get_class_definition(TypeshedClass('re', 'Pattern'))

re_pattern_anystr = instantiate_type_variables_in_class_definition(
    re_pattern_lookup_result,
    [anystr]
)

(
    re_pattern_anystr_match_parameter_type_annotation_list,
    re_pattern_anystr_match_return_value_type_annotation
) = get_comprehensive_type_annotations_for_parameters_and_return_values(
    re_pattern_anystr.method_name_to_method_list_dict['match']
)

assert re_pattern_anystr_match_return_value_type_annotation == Union(
    frozenset([
        Subscription(
            TypeshedClass('re', 'Match'),
            (
                TypeshedClass('builtins', 'str'),
            )
        ),
        Subscription(
            TypeshedClass('re', 'Match'),
            (
                TypeshedClass('builtins', 'bytes'),
            )
        ),
        TypeshedClass('types', 'NoneType')
    ])
)

assert re_pattern_anystr_match_parameter_type_annotation_list[2] == TypeshedClass('builtins', 'int')

builtins_list_definition = client.get_class_definition(TypeshedClass('builtins', 'list'))
type_variable = TypeshedTypeVariable()
subscribed_builtins_list_definition = instantiate_type_variables_in_class_definition(
    builtins_list_definition,
    (type_variable,)
)

(
    subscribed_builtins_list_append_parameter_type_annotation_list,
    subscribed_builtins_list_append_return_value_type_annotation
) = get_comprehensive_type_annotations_for_parameters_and_return_values(
    subscribed_builtins_list_definition.method_name_to_method_list_dict['append']
)

assert subscribed_builtins_list_append_parameter_type_annotation_list[1] == type_variable

(
    subscribed_builtins_list_extend_parameter_type_annotation_list,
    subscribed_builtins_list_extend_return_value_type_annotation
) = get_comprehensive_type_annotations_for_parameters_and_return_values(
    subscribed_builtins_list_definition.method_name_to_method_list_dict['extend']
)

assert subscribed_builtins_list_extend_parameter_type_annotation_list[1] == Subscription(
    TypeshedClass('typing', 'Iterable'),
    (
        type_variable,
    )
)

builtins_getattr_lookup_result = client.look_up_name('builtins', 'getattr')
assert builtins_getattr_lookup_result == TypeshedFunction('builtins', 'getattr')
builtins_getattr_function_definition = client.get_function_definition(builtins_getattr_lookup_result)
(
    builtins_getattr_parameter_type_annotation_list,
    builtins_getattr_return_value_type_annotation
) = get_comprehensive_type_annotations_for_parameters_and_return_values(
    builtins_getattr_function_definition
)

assert builtins_getattr_parameter_type_annotation_list[0] == TypeshedClass('builtins', 'object')
assert builtins_getattr_parameter_type_annotation_list[1] == TypeshedClass('builtins', 'str')

builtins_str_definition = client.get_class_definition(TypeshedClass('builtins', 'str'))

(
    builtins_str_split_parameter_type_annotation_list,
    builtins_str_split_return_value_type_annotation
) = get_comprehensive_type_annotations_for_parameters_and_return_values(
    builtins_str_definition.method_name_to_method_list_dict['split']
)

assert builtins_str_split_parameter_type_annotation_list[1] == Union(
    frozenset([
        TypeshedClass('types', 'NoneType'),
        TypeshedClass('builtins', 'str')
    ])
)

assert builtins_str_split_return_value_type_annotation == Subscription(
    TypeshedClass('builtins', 'list'),
    (
        TypeshedClass('builtins', 'str'),
    )
)

typing_coroutine_definition = client.get_class_definition(TypeshedClass('typing', 'Coroutine'))

(
    typing_coroutine_send_parameter_type_annotation_list,
    typing_coroutine_send_return_value_type_annotation
) = get_comprehensive_type_annotations_for_parameters_and_return_values(
    typing_coroutine_definition.method_name_to_method_list_dict['send']
)

assert typing_coroutine_send_return_value_type_annotation == typing_coroutine_definition.type_variable_list[0]

assert typing_coroutine_send_parameter_type_annotation_list[1] == typing_coroutine_definition.type_variable_list[1]


for (
        (runtime_class_module_name, runtime_class_name),
        (typeshed_class_module_name, typeshed_class_name)
) in runtime_class_name_tuple_to_typeshed_class_name_tuple.items():
    typeshed_class = TypeshedClass(typeshed_class_module_name, typeshed_class_name)
    runtime_class = to_runtime_class(typeshed_class)
    assert runtime_class is not None
    assert runtime_class_module_name == runtime_class.__module__
    assert runtime_class_name == runtime_class.__name__
    assert from_runtime_class(runtime_class) == typeshed_class

for (
    (typeshed_class_module_name, typeshed_class_name),
    runtime_class
) in typeshed_class_name_tuple_to_runtime_class.items():
    typeshed_class = from_runtime_class(runtime_class)
    assert typeshed_class.module_name == typeshed_class_module_name
    assert typeshed_class.class_name == typeshed_class_name
    new_runtime_class = to_runtime_class(typeshed_class)
    assert new_runtime_class is runtime_class

typing_iterable_definition = client.get_class_definition(TypeshedClass('typing', 'Iterable'))

assert len(typing_iterable_definition.type_variable_list) == 1

typing_iterator_definition = client.get_class_definition(TypeshedClass('typing', 'Iterator'))

assert len(typing_iterator_definition.type_variable_list) == 1
