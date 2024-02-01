import _ast
import argparse
import ast
import asyncio
import importlib
import json
import os
import sys
import typing
from itertools import chain
from types import ModuleType

import definitions_to_runtime_terms_mappings_singleton
import module_names_to_imported_names_to_runtime_objects_singleton
import node_textual_representation_singleton
import parameter_lists_and_symbolic_return_values_singleton
import static_import_analysis
import switches_singleton
import typing_constraints_singleton
from build_ast_node_namespace_trie import build_ast_node_namespace_trie
from class_query_database import ClassQueryDatabase
from collect_and_resolve_typing_constraints import collect_and_resolve_typing_constraints
from query_result_dict import *
from trie import search
from type_inference import TypeInference


# search ast_node_namespace_trie

def search_ast_node_namespace_trie(
    ast_node_namespace_trie_root,
    components: typing.Sequence[str]
) -> set[ast.AST]:
    if components:
        containing_namespace_components, name = components[:-1], components[-1]

        containing_namespace_trie_root = search(
            ast_node_namespace_trie_root,
            containing_namespace_components
        )

        if containing_namespace_trie_root is not None:
            if containing_namespace_trie_root.value is not None:
                if name in containing_namespace_trie_root.value:
                    return containing_namespace_trie_root.value[name]

    return set()

# get typing slots

def get_typing_slots_in_query_dict(query_dict):
    return list(chain.from_iterable(
        map(
            lambda module_name: get_typing_slots_in_module_level_query_dict(module_name, query_dict[module_name]),
            query_dict.keys(),
        )
    ))


def get_typing_slots_in_module_level_query_dict(module_name, module_level_query_dict):
    return list(chain.from_iterable(
        map(
            lambda class_name: get_typing_slots_in_class_level_query_dict(module_name, class_name, module_level_query_dict[class_name]),
            module_level_query_dict.keys(),
        )
    ))


def get_typing_slots_in_class_level_query_dict(module_name, class_name, class_level_query_dict):
    return list(chain.from_iterable(
        map(
            lambda function_name: get_typing_slots_in_functional_level_query_dict(module_name, class_name, function_name, class_level_query_dict[function_name]),
            class_level_query_dict.keys(),
        )
    ))


def get_typing_slots_in_functional_level_query_dict(module_name, class_name, function_name, functional_level_query_dict):
    return list(map(
        lambda parameter_name: [module_name, class_name, function_name, parameter_name],
        functional_level_query_dict,
    ))


def main():
    sys.setrecursionlimit(65536)

    # Set up logging
    # https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
    FORMAT = '[%(asctime)s %(filename)s %(funcName)s():%(lineno)s]%(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--module-search-path', type=str, required=True, help='Module search path, e.g. /tmp/module_search_path')
    parser.add_argument('-p', '--module-prefix', type=str, required=True, help="Module prefix")
    parser.add_argument('-o', '--output-file', type=str, required=False, default='output.json', help='Output JSON file')
    parser.add_argument('-i', '--interactive', action='store_true', required=False, default=False, help='Interactive mode')
    parser.add_argument('-q', '--query-dict', type=str, required=False, default=None, help='Query dict JSON file path')
    parser.add_argument('--class-inference-log-file', type=str, required=False, default=os.devnull)
    parser.add_argument('--class-attribute-csv-file', type=str, required=False, default=None, help='Export class-attribute matrix (document-term matrix) as a CSV file')
    parser.add_argument('--no-attribute-access-propagation', action='store_true', required=False)
    parser.add_argument('--no-stdlib-function-call-propagation', action='store_true', required=False)
    parser.add_argument('--no-user-defined-function-call-propagation', action='store_true', required=False)
    parser.add_argument('--no-shortcut-single-class-covering-all-attributes', action='store_true', required=False)
    parser.add_argument('--no-parameter-default-value-handling', action='store_true', required=False)
    parser.add_argument('--no-type-parameters', action='store_true', required=False)
    parser.add_argument('--log-term-frequency', action='store_true', required=False, default=False)
    parser.add_argument('--parameters-only', action='store_true', required=False, default=False)
    parser.add_argument('--return-values-only', action='store_true', required=False, default=False)
    parser.add_argument('--simplified-type-ascription', action='store_true', required=False, default=False)

    args = parser.parse_args()

    module_search_absolute_path: str = os.path.abspath(args.module_search_path)
    module_prefix: str = args.module_prefix

    # Find modules

    (
        module_name_to_file_path_dict,
        module_name_to_module_node_dict,
        module_name_to_function_name_to_parameter_name_list_dict,
        module_name_to_class_name_to_method_name_to_parameter_name_list_dict,
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict
    ) = static_import_analysis.do_static_import_analysis(module_search_absolute_path, module_prefix)

    node_textual_representation_singleton.initialize(
        module_name_to_module_node_dict
    )

    output_file: str = args.output_file
    
    is_interactive: bool = args.interactive
    
    # Read or generate query dict if not interactive

    if not is_interactive:
        if args.query_dict is not None:
            with open(args.query_dict, 'r') as fp:
                query_dict = json.load(fp)
        else:
            query_dict: QueryDict = generate_query_dict(
                module_name_to_file_path_dict,
                module_name_to_function_name_to_parameter_name_list_dict,
                module_name_to_class_name_to_method_name_to_parameter_name_list_dict
            )
    else:
        query_dict = {}

    # Set switches

    if args.no_attribute_access_propagation:
        switches_singleton.propagate_attribute_accesses = False

    if args.no_stdlib_function_call_propagation:
        switches_singleton.propagate_stdlib_function_calls = False

    if args.no_user_defined_function_call_propagation:
        switches_singleton.propagate_user_defined_function_calls = False

    if args.no_shortcut_single_class_covering_all_attributes:
        switches_singleton.shortcut_single_class_covering_all_attributes = False
    
    if args.no_parameter_default_value_handling:
        switches_singleton.handle_parameter_default_values = False

    if args.no_type_parameters:
        switches_singleton.predict_type_parameters = False

    if args.parameters_only:
        switches_singleton.parameters_only = True

    if args.return_values_only:
        switches_singleton.return_values_only = True
    
    if args.simplified_type_ascription:
        switches_singleton.simplified_type_ascription = True

    # Import modules
    sys.path.insert(0, module_search_absolute_path)

    module_name_to_module = {}
    for module_name, file_path in module_name_to_file_path_dict.items():
        try:
            module_name_to_module[module_name] = importlib.import_module(module_name)
        except ImportError:
            logging.exception('Failed to import module %s', module_name)

    # Initialize class query database

    module_set: set[ModuleType] = set(module_name_to_module.values())

    for module_name, import_tuple_set in module_name_to_import_tuple_set_dict.items():
        for imported_module_name, imported_module_name_alias in import_tuple_set:
            if imported_module_name in sys.modules:
                imported_module = sys.modules[imported_module_name]
                if isinstance(imported_module, ModuleType):
                    module_set.add(imported_module)

    for module_name, import_from_tuple_set in module_name_to_import_from_tuple_set_dict.items():
        for import_from_module_name, imported_name, imported_name_alias in import_from_tuple_set:
            if import_from_module_name in sys.modules:
                import_from_module = sys.modules[import_from_module_name]
                if isinstance(import_from_module, ModuleType):
                    module_set.add(import_from_module)

    class_query_database = ClassQueryDatabase(
        module_set,
        typing_constraints_singleton.client
    )

    if args.class_attribute_csv_file is not None:
        class_query_database.class_attribute_matrix.to_csv(args.class_attribute_csv_file)

    # Collect and resolve typing constraints

    module_name_list = list(module_name_to_module.keys())
    module_list = list(module_name_to_module.values())
    module_node_list = [module_name_to_module_node_dict[module_name] for module_name in module_name_to_module]

    parameter_lists_and_symbolic_return_values_singleton.get_parameter_lists_and_symbolic_return_values(
        module_node_list
    )

    ast_node_namespace_trie_root = build_ast_node_namespace_trie(
        module_name_list,
        module_node_list
    )

    import debugging_singleton

    definitions_to_runtime_terms_mappings_singleton.get_definitions_to_runtime_terms_mappings(
        module_name_list,
        module_list,
        module_node_list
    )

    module_names_to_imported_names_to_runtime_objects_singleton.get_module_names_to_imported_names_to_runtime_objects(
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict,
        sys.modules
    )

    asyncio.run(collect_and_resolve_typing_constraints(
        module_name_to_module_node_dict
    ))

    # Initialize type inference
    type_inference = TypeInference(
        class_query_database,
        typing_constraints_singleton.client
    )

    # Run type inference
    with open(args.class_inference_log_file, 'w') as class_inference_log_file_io:
        if is_interactive:
            while True:
                user_input = input('Enter the full containing namespace and name of the node (e.g. module_name class_name function_name name):')

                components = user_input.split()

                node_set = search_ast_node_namespace_trie(ast_node_namespace_trie_root, components)

                type_inference_result = type_inference.infer_type(
                    frozenset(node_set),
                    class_inference_log_file_io=class_inference_log_file_io
                )

                print('Type inference result:', type_inference_result, file=sys.stderr)

        else:
            raw_result_defaultdict: RawResultDefaultdict = get_raw_result_defaultdict()

            for (
                module_name_,
                class_name_or_global_,
                function_name_,
                parameter_name_or_return_
            ) in get_typing_slots_in_query_dict(query_dict):
                if class_name_or_global_ == 'global':
                    components = [module_name_, function_name_, parameter_name_or_return_]
                else:
                    components = [module_name_, class_name_or_global_, function_name_, parameter_name_or_return_]
                
                node_set = search_ast_node_namespace_trie(ast_node_namespace_trie_root, components)

                type_inference_result_list = raw_result_defaultdict[module_name_][class_name_or_global_][function_name_][parameter_name_or_return_]

                # Do not infer parameter types for self and cls in methods of classes.
                # Do not infer return types for __init__ and __new__ of classes.
                if (
                    (class_name_or_global_ != 'global' and parameter_name_or_return_ in ('self', 'cls'))
                    or (class_name_or_global_ != 'global' and function_name_ in ('__init__', '__new__') and parameter_name_or_return_ == 'return')
                ):
                    continue
                
                type_inference_result = type_inference.infer_type(
                    frozenset(node_set),
                    class_inference_log_file_io=class_inference_log_file_io
                )

                type_inference_result_list.append(str(type_inference_result))

            with open(output_file, 'w') as output_file_io:
                json.dump(raw_result_defaultdict, output_file_io, indent=4)


if __name__ == '__main__':
    main()
