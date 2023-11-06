import argparse
import ast
import readline
import importlib
import json
import os
import sys
import typing
from types import ModuleType

import static_import_analysis
from attribute_counter import AttributeCounter
from build_ast_node_namespace_trie import build_ast_node_namespace_trie
from class_query_database import ClassQueryDatabase
from collect_preliminary_typing_constraints import collect_preliminary_typing_constraints
from get_definitions_to_runtime_terms_mappings import get_definitions_to_runtime_terms_mappings
from get_top_level_imported_names_to_runtime_objects_mappings import \
    get_top_level_imported_names_to_runtime_objects_mappings
from query_result_dict import *
from relations import NonEquivalenceRelationGraph, EquivalenceRelationGraph
from resolve_typing_constraints import resolve_typing_constraints
from runtime_term import RuntimeTerm
from trie import search
from type_inference import ClassInference
from typeshed_client_ex.client import Client
from typeshed_client_ex.type_definitions import TypeshedClass, from_runtime_class


def get_indentation(indent_level: int = 0) -> str:
    return '    ' * indent_level


def node_to_string(node: ast.AST, node_to_module_name_dict: dict[ast.AST, str]) -> str:
    return ' '.join([
        str(node),
        ast.unparse(node),
        str(node_to_module_name_dict.get(node, None)),
        str(getattr(node, 'lineno', None)),
    ])


def print_equivalent_sets(
        equivalent_sets: typing.Iterable[set[ast.AST]],
        top_nodes: typing.Iterable[ast.AST],
        nodes_to_attribute_counters: defaultdict[ast.AST, AttributeCounter],
        equivalence_relation_graph: EquivalenceRelationGraph,
        type_variable_relations: NonEquivalenceRelationGraph,
        equivalent_type_variable_attribute_counters: defaultdict[ast.AST, AttributeCounter],
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
            nodes_to_attribute_counters,
            equivalence_relation_graph,
            type_variable_relations,
            equivalent_type_variable_attribute_counters,
            equivalent_type_variable_runtime_term_sets,
            node_to_module_name_dict,
            indent_level + 1
        )


def print_equivalent_set(
        equivalent_set: set[ast.AST],
        top_node: ast.AST,
        nodes_to_attribute_counters: defaultdict[ast.AST, AttributeCounter],
        equivalence_relation_graph: EquivalenceRelationGraph,
        type_variable_relations: NonEquivalenceRelationGraph,
        equivalent_type_variable_attribute_counters: defaultdict[ast.AST, AttributeCounter],
        equivalent_type_variable_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]],
        node_to_module_name_dict: dict[ast.AST, str],
        indent_level: int = 0
):
    print(get_indentation(indent_level), 'Nodes:', file=sys.stderr)
    for node in equivalent_set:
        print_node(
            node,
            nodes_to_attribute_counters,
            equivalence_relation_graph,
            type_variable_relations,
            node_to_module_name_dict,
            indent_level + 1
        )

    print(get_indentation(indent_level), 'Equivalence relations:', file=sys.stderr)
    for first_node, second_node in equivalence_relation_graph.get_equivalence_relations_among_nodes(equivalent_set):
        print(get_indentation(indent_level + 1), node_to_string(first_node, node_to_module_name_dict), '->',
              node_to_string(second_node, node_to_module_name_dict), file=sys.stderr)

    print(get_indentation(indent_level), 'Collective attribute counter:', file=sys.stderr)
    print_attribute_counter(
        equivalent_type_variable_attribute_counters[top_node],
        indent_level + 1
    )

    print(get_indentation(indent_level), 'Collective runtime terms:', file=sys.stderr)
    print_runtime_term_set(
        equivalent_type_variable_runtime_term_sets[top_node],
        indent_level + 1
    )


def print_node(
        node: ast.AST,
        nodes_to_attribute_counters: defaultdict[ast.AST, AttributeCounter],
        equivalence_relation_graph: EquivalenceRelationGraph,
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
        nodes_to_attribute_counters[node],
        indent_level + 2
    )
    print(get_indentation(indent_level + 1), 'Related nodes:', file=sys.stderr)
    print_related_nodes(node, nodes_to_attribute_counters, equivalence_relation_graph, type_variable_relations, node_to_module_name_dict,
                        indent_level + 2,
                        exclude | frozenset([node]))


def print_attribute_counter(
        attribute_counter: AttributeCounter,
        indent_level: int = 0
):
    for key, value in attribute_counter.items():
        print(get_indentation(indent_level), key, value, file=sys.stderr)


def print_runtime_term_set(
        runtime_term_set: set[RuntimeTerm],
        indent_level: int = 0
):
    for runtime_term in runtime_term_set:
        print(get_indentation(indent_level), runtime_term, file=sys.stderr)


def print_related_nodes(
        node: ast.AST,
        nodes_to_attribute_counters: defaultdict[ast.AST, AttributeCounter],
        equivalence_relation_graph: EquivalenceRelationGraph,
        type_variable_relations: NonEquivalenceRelationGraph,
        node_to_module_name_dict: dict[ast.AST, str],
        indent_level: int = 0,
        exclude: frozenset[ast.AST] = frozenset()
):
    for relation_tuple, related_node_set in type_variable_relations.get_out_edges_by_relation_tuple(node).items():
        print(get_indentation(indent_level), relation_tuple, file=sys.stderr)
        for related_node in related_node_set:
            print_node(related_node, nodes_to_attribute_counters, equivalence_relation_graph, type_variable_relations, node_to_module_name_dict,
                       indent_level + 1, exclude)


def main():
    # Set up logging
    # https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
    FORMAT = '[%(asctime)s %(filename)s %(funcName)s():%(lineno)s]%(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--module-search-path', type=str, required=True,
                        help='Module search path, e.g. /tmp/module_search_path')
    parser.add_argument('-o', '--output-file', type=str, required=False,
                        default='output.json',
                        help='Output JSON file')
    parser.add_argument('-i', '--interactive', action='store_true', required=False, default=False,
                        help='Interactive mode')
    args = parser.parse_args()

    module_search_absolute_path: str = os.path.abspath(args.module_search_path)
    output_file: str = args.output_file
    is_interactive: bool = args.interactive

    # Initialize Typeshed client
    client: Client = Client()

    # Find modules
    (
        module_name_to_file_path_dict,
        module_name_to_module_node_dict,
        module_name_to_function_name_to_parameter_name_list_dict,
        module_name_to_class_name_to_method_name_to_parameter_name_list_dict,
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict
    ) = static_import_analysis.do_static_import_analysis(module_search_absolute_path)

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

    (
        top_level_class_definitions_to_runtime_classes,
        unwrapped_runtime_functions_to_named_function_definitions
    ) = get_definitions_to_runtime_terms_mappings(
        module_name_list,
        module_list,
        module_node_list
    )

    module_names_to_imported_names_to_runtime_objects = get_top_level_imported_names_to_runtime_objects_mappings(
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict,
        sys.modules
    )

    (
        nodes_to_attribute_counters,
        nodes_to_runtime_term_sets,
        nodes_providing_scope_to_parameter_lists_and_return_value_sets,
        equivalence_relation_graph,
        non_equivalence_relation_graph
    ) = collect_preliminary_typing_constraints(
        top_level_class_definitions_to_runtime_classes,
        unwrapped_runtime_functions_to_named_function_definitions,
        module_names_to_imported_names_to_runtime_objects,
        module_name_to_module_node_dict
    )

    (
        equivalent_node_disjoint_set,
        new_equivalence_relation_graph,
        new_non_equivalence_relation_graph,
        new_nodes_to_attribute_counters,
        equivalent_node_disjoint_set_top_nodes_to_attribute_counters,
        equivalent_node_disjoint_set_top_nodes_to_runtime_term_sets,
        equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph
    ) = resolve_typing_constraints(
        unwrapped_runtime_functions_to_named_function_definitions,
        nodes_to_attribute_counters,
        nodes_to_runtime_term_sets,
        nodes_providing_scope_to_parameter_lists_and_return_value_sets,
        equivalence_relation_graph,
        non_equivalence_relation_graph,
        client
    )

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
        client
    )

    # Initialize type inference
    type_inference = ClassInference(
        equivalent_node_disjoint_set_top_nodes_to_attribute_counters,
        equivalent_node_disjoint_set_top_nodes_to_runtime_term_sets,
        equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph,
        class_query_database,
        client
    )

    top_node_to_equivalent_set: defaultdict[ast.AST, set[ast.AST]] = defaultdict(set)
    for top_node, equivalent_set in equivalent_node_disjoint_set.itersets():
        top_node_to_equivalent_set[top_node] = equivalent_set

    # Run type inference

    node_to_module_name_dict: dict[ast.AST, str] = dict()
    for module_name, module_node in module_name_to_module_node_dict.items():
        for node in ast.walk(module_node):
            node_to_module_name_dict[node] = module_name

    def class_inference_print_equivalent_set_callback(top_node_frozenset: frozenset[ast.AST], indent_level: int):
        top_node_list = list(top_node_frozenset)

        equivalent_set_list = [
            top_node_to_equivalent_set[top_node]
            for top_node in top_node_list
        ]

        print_equivalent_sets(
            equivalent_set_list,
            top_node_list,
            new_nodes_to_attribute_counters,
            new_equivalence_relation_graph,
            new_non_equivalence_relation_graph,
            equivalent_node_disjoint_set_top_nodes_to_attribute_counters,
            equivalent_node_disjoint_set_top_nodes_to_runtime_term_sets,
            node_to_module_name_dict,
            indent_level + 1
        )

    if is_interactive:
        ast_node_namespace_trie_root = build_ast_node_namespace_trie(
            module_name_list,
            module_node_list
        )

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
                equivalent_node_disjoint_set.find(node)
                for node in node_set
            })

            type_inference_result = type_inference.infer_type_for_equivalent_node_disjoint_set_top_nodes(
                top_node_frozenset,
                class_inference_callback=class_inference_print_equivalent_set_callback
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

                    parameter_list, return_value_set = nodes_providing_scope_to_parameter_lists_and_return_value_sets[
                        function_node
                    ]

                    parameter_names_to_parameter_node_sets: dict[str, set[ast.AST]] = {
                        'return': return_value_set
                    }

                    for parameter in parameter_list:
                        parameter_names_to_parameter_node_sets[parameter.arg] = {parameter}

                    for parameter_name, parameter_node_set in parameter_names_to_parameter_node_sets.items():
                        logging.info(
                            'Inferring types for %s::%s::%s::%s',
                            module_name,
                            class_name,
                            function_name,
                            parameter_name
                        )

                        top_node_frozenset = frozenset({
                            equivalent_node_disjoint_set.find(node)
                            for node in parameter_node_set
                        })

                        type_inference_result = type_inference.infer_type_for_equivalent_node_disjoint_set_top_nodes(
                            top_node_frozenset,
                            class_inference_callback=class_inference_print_equivalent_set_callback,
                            first_level_class_inference_failed_fallback=(
                                from_runtime_class(type(None))
                                if parameter_name == 'return'
                                else TypeshedClass('typing', 'Any')
                            )
                        )

                        raw_result_defaultdict[module_name][class_name][function_name][parameter_name].append(
                            str(type_inference_result)
                        )

        with open(output_file, 'w') as fp:
            json.dump(raw_result_defaultdict, fp, indent=4)


if __name__ == '__main__':
    main()
