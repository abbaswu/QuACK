from typing import Callable

from lark import Lark, Token, Tree

from type_inference_class import TypeInferenceClass
from type_inference_result import TypeInferenceResult


parser: Lark = Lark(r"""
type_annotation: class | subscription | callable
class: NAME ("." NAME)*
subscription: class "[" type_annotation ("," type_annotation)* "]"
callable: "def" "(" arguments ")" "->" type_annotation
arguments: arg ("," arg)* | vararg | 
arg: NAME ":" type_annotation
vararg: "*" NAME ":" type_annotation

%import common.WS
%import python.NAME

%ignore WS
""",
start='type_annotation',
parser='lalr')


def handle_type_annotation_tree(
    type_annotation_tree: Tree,
    last_module_component_and_class_name_to_class_dict: dict[tuple[str, str], TypeInferenceClass]
) -> TypeInferenceResult:
    # type_annotation: class | subscription | callable
    class_subscription_or_callable_tree: Tree = type_annotation_tree.children[0]
    rule: str = class_subscription_or_callable_tree.data.value
    if rule == 'class':
        type_inference_class: TypeInferenceClass = handle_class_tree(
            class_subscription_or_callable_tree,
            last_module_component_and_class_name_to_class_dict
        )
        return TypeInferenceResult(type_inference_class)
    elif rule == 'subscription':
        return handle_subscription_tree(
            class_subscription_or_callable_tree,
            last_module_component_and_class_name_to_class_dict
        )
    elif rule == 'callable':
        return handle_callable_tree(
            class_subscription_or_callable_tree,
            last_module_component_and_class_name_to_class_dict
        )
    else:
        assert False


def handle_class_tree(
    class_tree: Tree,
    last_module_component_and_class_name_to_class_dict: dict[tuple[str, str], TypeInferenceClass]
) -> TypeInferenceClass:
    # class: NAME ("." NAME)*
    names: list[str] = [ child.value for child in class_tree.children ]
    if len(names) == 1:
        name: str = names[0]
        # Resolve None to builtins.NoneType
        if name == 'None':
            return TypeInferenceClass('builtins', 'NoneType')
        # Resolve Any to typing.Any
        elif name == 'Any':
            return TypeInferenceClass('typing', 'Any')
        # Resolve Tuple to builtins.tuple
        elif name == 'Tuple':
            return TypeInferenceClass('builtins', 'tuple')
        else:
            assert False
    elif len(names) == 2:
        last_module_component: str = names[0]
        class_name: str = names[1]
        return last_module_component_and_class_name_to_class_dict.get(
            (last_module_component, class_name),
            TypeInferenceClass(last_module_component, class_name)
        )
    elif len(names) > 2:
        module_name: str = '.'.join(names[:-1])
        class_name: str = names[-1]
        return TypeInferenceClass(module_name, class_name)
    else:
        assert False


def handle_subscription_tree(
    subscription_tree: Tree,
    last_module_component_and_class_name_to_class_dict: dict[tuple[str, str], TypeInferenceClass]
) -> TypeInferenceResult:
    # subscription: class "[" type_annotation ("," type_annotation)* "]"
    class_tree: Tree = subscription_tree.children[0]
    type_inference_class: TypeInferenceClass = handle_class_tree(
        class_tree,
        last_module_component_and_class_name_to_class_dict
    )
    filled_type_variable_list: list[TypeInferenceResult] = [
        handle_type_annotation_tree(
            type_annotation_tree,
            last_module_component_and_class_name_to_class_dict
        )
        for type_annotation_tree in subscription_tree.children[1:]
    ]
    # Special handling for `builtins.tuple`'s with 1 filled type variable
    if type_inference_class == TypeInferenceClass('builtins', 'tuple') and len(filled_type_variable_list) == 1:
        filled_type_variable_list.append(TypeInferenceResult(TypeInferenceClass('builtins', 'ellipsis')))
    return TypeInferenceResult(type_inference_class, tuple(filled_type_variable_list))


def handle_callable_tree(
    callable_tree: Tree,
    last_module_component_and_class_name_to_class_dict: dict[tuple[str, str], TypeInferenceClass]
) -> TypeInferenceResult:
    # "def" "(" arguments ")" "->" type_annotation
    arguments_tree: Tree = callable_tree.children[0]
    type_annotation_tree: Tree = callable_tree.children[1]
    parameters_and_return_value_type_inference_result_list: list[
        TypeInferenceResult
    ] = []
    parameters_and_return_value_type_inference_result_list.extend(
        handle_arguments_tree(
            arguments_tree,
            last_module_component_and_class_name_to_class_dict
        )
    )
    parameters_and_return_value_type_inference_result_list.append(
        handle_type_annotation_tree(
            type_annotation_tree,
            last_module_component_and_class_name_to_class_dict
        )
    )
    return TypeInferenceResult(
        TypeInferenceClass('typing', 'Callable'),
        tuple(parameters_and_return_value_type_inference_result_list)
    )


def handle_arguments_tree(
    arguments_tree: Tree,
    last_module_component_and_class_name_to_class_dict: dict[tuple[str, str], TypeInferenceClass]
) -> list[TypeInferenceResult]:
    # arguments: arg ("," arg)* | vararg |
    parameters_type_inference_result_list: list[TypeInferenceResult] = []
    if arguments_tree.children:
        rule: str = arguments_tree.children[0].data.value
        if rule == 'arg':
            for arg_tree in arguments_tree.children:
                parameters_type_inference_result_list.append(
                    handle_arg_or_vararg_tree(
                        arg_tree,
                        last_module_component_and_class_name_to_class_dict
                    )
                )
        elif rule == 'vararg':
            # Use a single builtins.ellipsis to represent varargs
            parameters_type_inference_result_list.append(
                TypeInferenceResult(TypeInferenceClass('builtins', 'ellipsis'))
            )
    return parameters_type_inference_result_list


def handle_arg_or_vararg_tree(
    arg_or_vararg_tree: Tree,
    last_module_component_and_class_name_to_class_dict: dict[tuple[str, str], TypeInferenceClass]
) -> TypeInferenceResult:
    # arg: NAME ":" type_annotation
    # vararg: "*" NAME ":" type_annotation
    type_annotation_tree: Tree = arg_or_vararg_tree.children[1]
    return handle_type_annotation_tree(
        type_annotation_tree,
        last_module_component_and_class_name_to_class_dict
    )


def get_type_annotation_parser(
    module_name_to_class_name_to_method_name_to_parameter_name_list_dict: dict[str, dict[str, dict[str, list[str]]]],
    module_name_to_import_from_tuple_set_dict: dict[str, set[tuple[str, str, str]]]
) -> Callable[[str, str], TypeInferenceResult]:
    # Construct last_module_component_and_class_name_to_class_dict
    last_module_component_and_class_name_to_class_dict: dict[tuple[str, str], TypeInferenceClass] = dict()

    for module_name, class_name_to_method_name_to_parameter_name_list_dict in module_name_to_class_name_to_method_name_to_parameter_name_list_dict.items():
        module_components: list[str] = module_name.split('.')
        last_module_component: str = module_components[-1]

        for class_name in class_name_to_method_name_to_parameter_name_list_dict:
            type_inference_class: TypeInferenceClass = TypeInferenceClass(module_name, class_name)

            last_module_component_and_class_name_to_class_dict[(last_module_component, class_name)] = type_inference_class

    # Define type_annotation_parser
    def type_annotation_parser(
        module_name: str,
        type_annotation_string: str
    ) -> TypeInferenceResult:
        nonlocal last_module_component_and_class_name_to_class_dict

        type_annotation_tree: Tree = parser.parse(type_annotation_string)
        return handle_type_annotation_tree(
            type_annotation_tree,
            last_module_component_and_class_name_to_class_dict
        )

    # Return type_annotation_parser
    return type_annotation_parser
