import _ast
import argparse
import ast
import asyncio
import importlib
import json
import os
import sys
import typing
from collections import Counter
from types import ModuleType

import definitions_to_runtime_terms_mappings_singleton
import module_names_to_imported_names_to_runtime_objects_singleton
import parameter_lists_and_symbolic_return_values_singleton
import static_import_analysis
import switches_singleton
import typing_constraints_singleton
from build_ast_node_namespace_trie import build_ast_node_namespace_trie
from class_query_database import ClassQueryDatabase
from collect_and_resolve_typing_constraints import collect_and_resolve_typing_constraints
from query_result_dict import *
from relations import NonEquivalenceRelationGraph
from runtime_term import RuntimeTerm
from trie import search
from type_inference import TypeInference
from typing_constraints_singleton import node_disjoint_set, \
    equivalent_set_top_node_non_equivalence_relation_graph, nodes_to_attribute_counters, \
    node_non_equivalence_relation_graph, runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets, \
    runtime_term_sharing_node_disjoint_set


def get_indentation(indent_level: int = 0) -> str:
    return '    ' * indent_level


def node_to_string(node: ast.AST, node_to_module_name_dict: dict[ast.AST, str]) -> str:
    return ': '.join([
        str(node),
        str(node_to_module_name_dict.get(node, None)),
        str(getattr(node, 'lineno', None)),
        ast.unparse(node),
    ])


def print_equivalent_sets(
        equivalent_sets: typing.Iterable[typing.Set[ast.AST]],
        top_nodes: typing.Iterable[ast.AST],
        node_attribute_counter: defaultdict[_ast.AST, Counter[str]],
        type_variable_relations: NonEquivalenceRelationGraph,
        equivalent_type_variable_attribute_counters: defaultdict[ast.AST, Counter[str]],
        equivalent_type_variable_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]],
        node_to_module_name_dict: dict[ast.AST, str],
        indent_level: int = 0
):
    for i, (equivalent_set, top_node) in enumerate(
            zip(
                equivalent_sets,
                top_nodes
            )
    ):
        print(get_indentation(indent_level), f'Equivalent set {i}:', file=sys.stderr)
        print_equivalent_set(
            equivalent_set,
            top_node,
            node_attribute_counter,
            type_variable_relations,
            equivalent_type_variable_attribute_counters,
            equivalent_type_variable_runtime_term_sets,
            node_to_module_name_dict,
            indent_level + 1
        )


def print_equivalent_set(
        equivalent_set: typing.Set[ast.AST],
        top_node: ast.AST,
        node_attribute_counter: defaultdict[_ast.AST, Counter[str]],
        type_variable_relations: NonEquivalenceRelationGraph,
        equivalent_type_variable_attribute_counters: defaultdict[ast.AST, Counter[str]],
        equivalent_type_variable_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]],
        node_to_module_name_dict: dict[ast.AST, str],
        indent_level: int = 0
):
    print(get_indentation(indent_level), 'Nodes:', file=sys.stderr)
    for node in equivalent_set:
        print_node(
            node,
            node_attribute_counter,
            type_variable_relations,
            node_to_module_name_dict,
            indent_level + 1
        )

    print(get_indentation(indent_level), 'Collective runtime terms:', file=sys.stderr)
    print_runtime_term_set(
        equivalent_type_variable_runtime_term_sets[top_node],
        indent_level + 1
    )


def print_node(
        node: ast.AST,
        node_attribute_counter: defaultdict[_ast.AST, Counter[str]],
        type_variable_relations: NonEquivalenceRelationGraph,
        node_to_module_name_dict,
        indent_level: int = 0,
        exclude: frozenset[ast.AST] = frozenset()
):
    if node in exclude:
        return

    print(get_indentation(indent_level), node, file=sys.stderr)
    print(get_indentation(indent_level + 1), 'Representation:', node_to_string(node, node_to_module_name_dict),
          file=sys.stderr)
    print(get_indentation(indent_level + 1), 'Attribute counter:', file=sys.stderr)
    print_attribute_counter(
        node,
        node_attribute_counter,
        indent_level + 2
    )
    print(get_indentation(indent_level + 1), 'Related nodes:', file=sys.stderr)
    print_related_nodes(node, node_attribute_counter, type_variable_relations, node_to_module_name_dict,
                        indent_level + 2,
                        exclude | frozenset([node]))


def print_attribute_counter(
        node: _ast.AST,
        node_attribute_counter: defaultdict[_ast.AST, Counter[str]],
        indent_level: int = 0
):
    for key, value in node_attribute_counter[node].items():
        print(get_indentation(indent_level), key, value, file=sys.stderr)


def print_runtime_term_set(
        runtime_term_set: set[RuntimeTerm],
        indent_level: int = 0
):
    for runtime_term in runtime_term_set:
        print(get_indentation(indent_level), runtime_term, file=sys.stderr)


def print_related_nodes(
        node: ast.AST,
        node_attribute_counter: defaultdict[_ast.AST, Counter[str]],
        type_variable_relations: NonEquivalenceRelationGraph,
        node_to_module_name_dict: dict[ast.AST, str],
        indent_level: int = 0,
        exclude: frozenset[ast.AST] = frozenset()
):
    for relation_type, parameter_to_related_node_set in type_variable_relations.get_out_nodes(node).items():
        for parameter, related_node_set in parameter_to_related_node_set.items():
            print(get_indentation(indent_level), relation_type, parameter, file=sys.stderr)
            for related_node in related_node_set:
                print_node(related_node, node_attribute_counter, type_variable_relations, node_to_module_name_dict,
                           indent_level + 1, exclude)


def contains_return_statement(function_node: typing.Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
    return any(
        isinstance(node, ast.Return)
        for node in ast.walk(function_node)
    )


def main():
    sys.setrecursionlimit(65536)

    # Set up logging
    # https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
    FORMAT = '[%(asctime)s %(filename)s %(funcName)s():%(lineno)s]%(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--module-search-path', type=str, required=True,
                        help='Module search path, e.g. /tmp/module_search_path')
    parser.add_argument('-p', '--module-prefix', type=str, required=True,
                        help="Module prefix")
    parser.add_argument('-o', '--output-file', type=str, required=False,
                        default='output.json',
                        help='Output JSON file')
    parser.add_argument('-i', '--interactive', action='store_true', required=False, default=False,
                        help='Interactive mode')
    parser.add_argument('--class-inference-log-file', type=str, required=False, default=os.devnull)
    parser.add_argument('--no-induced-equivalent-relation-resolution', action='store_true', required=False)
    parser.add_argument('--no-attribute-access-propagation', action='store_true', required=False)
    parser.add_argument('--no-stdlib-function-call-propagation', action='store_true', required=False)
    parser.add_argument('--no-user-defined-function-call-propagation', action='store_true', required=False)

    args = parser.parse_args()

    module_search_absolute_path: str = os.path.abspath(args.module_search_path)
    module_prefix: str = args.module_prefix
    output_file: str = args.output_file
    is_interactive: bool = args.interactive

    # Set switches
    if args.no_induced_equivalent_relation_resolution:
        switches_singleton.resolve_induced_equivalent_relations = False

    if args.no_attribute_access_propagation:
        switches_singleton.propagate_attribute_accesses = False

    if args.no_stdlib_function_call_propagation:
        switches_singleton.propagate_stdlib_function_calls = False

    if args.no_user_defined_function_call_propagation:
        switches_singleton.propagate_user_defined_function_calls = False

    # Find modules
    (
        module_name_to_file_path_dict,
        module_name_to_module_node_dict,
        module_name_to_function_name_to_parameter_name_list_dict,
        module_name_to_class_name_to_method_name_to_parameter_name_list_dict,
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict
    ) = static_import_analysis.do_static_import_analysis(module_search_absolute_path, module_prefix)

    # Import modules
    sys.path.insert(0, module_search_absolute_path)

    module_name_to_module = {
        module_name: importlib.import_module(module_name)
        for module_name, file_path in module_name_to_file_path_dict.items()
    }

    # Collect and resolve typing constraints
    module_name_list = list(module_name_to_module.keys())
    module_list = list(module_name_to_module.values())
    module_node_list = [module_name_to_module_node_dict[module_name] for module_name in module_name_to_module]

    ast_node_namespace_trie_root = build_ast_node_namespace_trie(
        module_name_list,
        module_node_list
    )

    parameter_lists_and_symbolic_return_values_singleton.get_parameter_lists_and_symbolic_return_values(
        module_node_list
    )

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

    # Initialize type inference
    equivalent_set_top_nodes_to_attribute_counters: defaultdict[_ast.AST, Counter[str]] = defaultdict(Counter)
    for equivalent_set_top_node, equivalent_set in node_disjoint_set.itersets():
        for node in equivalent_set:
            equivalent_set_top_nodes_to_attribute_counters[equivalent_set_top_node] += nodes_to_attribute_counters[node]

    type_inference = TypeInference(
        node_disjoint_set,
        equivalent_set_top_nodes_to_attribute_counters,
        equivalent_set_top_node_non_equivalence_relation_graph,
        runtime_term_sharing_node_disjoint_set,
        runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets,
        class_query_database,
        typing_constraints_singleton.client
    )

    # Run type inference

    node_to_module_name_dict: dict[ast.AST, str] = dict()
    for module_name, module_node in module_name_to_module_node_dict.items():
        for node in ast.walk(module_node):
            node_to_module_name_dict[node] = module_name

    def pre_class_inference_print_equivalent_set_callback(top_node_frozenset: frozenset[ast.AST], indent_level: int):
        top_node_list = list(top_node_frozenset)

        equivalent_set_list = [
            node_disjoint_set.get_containing_set(top_node)
            for top_node in top_node_list
        ]

        print_equivalent_sets(
            equivalent_set_list,
            top_node_list,
            nodes_to_attribute_counters,
            node_non_equivalence_relation_graph,
            equivalent_set_top_nodes_to_attribute_counters,
            runtime_term_sharing_equivalent_set_top_nodes_to_runtime_term_sets,
            node_to_module_name_dict,
            indent_level + 1
        )

    with open(args.class_inference_log_file, 'w') as class_inference_log_file_io:
        if is_interactive:
            while True:
                user_input = input(
                    'Enter the full containing namespace and name of the node (e.g. module_name class_name function_name name):')
                components = user_input.split()

                if not components:
                    continue

                containing_namespace_components, name = components[:-1], components[-1]

                containing_namespace_trie_root = search(
                    ast_node_namespace_trie_root,
                    containing_namespace_components
                )

                if containing_namespace_trie_root is None:
                    continue

                if containing_namespace_trie_root.value is None:
                    continue

                if name not in containing_namespace_trie_root.value:
                    continue

                node_set = containing_namespace_trie_root.value[name]

                top_node_frozenset = frozenset({
                    node_disjoint_set.find(node)
                    for node in node_set
                })

                type_inference_result = type_inference.infer_type_for_equivalent_node_disjoint_set_top_nodes(
                    top_node_frozenset,
                    pre_class_inference_callback=pre_class_inference_print_equivalent_set_callback,
                    class_inference_log_file_io=class_inference_log_file_io
                )

                print('Type inference result:', type_inference_result, file=sys.stderr)

        else:
            raw_result_defaultdict: RawResultDefaultdict = get_raw_result_defaultdict()

            for module_name, module_node in module_name_to_module_node_dict.items():
                class_names_to_function_node_sets: dict[
                    str,
                    set[typing.Union[ast.FunctionDef, ast.AsyncFunctionDef]]
                ] = {
                    'global': set()
                }

                for node in module_node.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_names_to_function_node_sets['global'].add(node)
                    elif isinstance(node, ast.ClassDef):
                        class_name = node.name
                        function_node_set: set[typing.Union[ast.FunctionDef, ast.AsyncFunctionDef]] = set()

                        for child_node in node.body:
                            if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                function_node_set.add(child_node)

                        class_names_to_function_node_sets[class_name] = function_node_set

                for class_name, function_node_set in class_names_to_function_node_sets.items():
                    for function_node in function_node_set:
                        function_name = function_node.name

                        parameter_list, symbolic_return_value = \
                        parameter_lists_and_symbolic_return_values_singleton.nodes_to_parameter_lists_and_symbolic_return_values[
                            function_node
                        ]

                        parameter_names_to_parameter_nodes: dict[str, _ast.AST] = {}

                        # Only infer return value types for functions that are:
                        # Not __init__ and __new__ of classes, and,
                        # Contain return statements.
                        if (
                                not (class_name != 'global' and function_name in ('__init__', '__new__'))
                                and contains_return_statement(function_node)
                        ):
                            parameter_names_to_parameter_nodes['return'] = symbolic_return_value

                        # Do not infer parameter types for self and cls in methods of classes.
                        for i, parameter in enumerate(parameter_list):
                            if not (class_name != 'global' and i == 0 and parameter.arg in ('self', 'cls')):
                                parameter_names_to_parameter_nodes[parameter.arg] = parameter

                        for parameter_name, parameter_node in parameter_names_to_parameter_nodes.items():
                            logging.info(
                                'Inferring types for %s::%s::%s::%s',
                                module_name,
                                class_name,
                                function_name,
                                parameter_name
                            )

                            top_node = node_disjoint_set.find(parameter_node)

                            type_inference_result = type_inference.infer_type_for_equivalent_node_disjoint_set_top_nodes(
                                frozenset([top_node]),
                                pre_class_inference_callback=pre_class_inference_print_equivalent_set_callback,
                                class_inference_log_file_io=class_inference_log_file_io
                            )

                            raw_result_defaultdict[module_name][class_name][function_name][parameter_name].append(
                                str(type_inference_result)
                            )

            with open(output_file, 'w') as output_file_io:
                json.dump(raw_result_defaultdict, output_file_io, indent=4)


if __name__ == '__main__':
    main()
