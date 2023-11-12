"""
coverage run --source=collect_preliminary_typing_constraints,resolve_typing_constraints test_resolve_typing_constraints.py

"""
import ast
import collections.abc
import importlib
import json
import sys
import typing

from collections import defaultdict, Counter

import networkx as nx

from attribute_counter import AttributeCounter
import static_import_analysis
from build_ast_node_namespace_trie import build_ast_node_namespace_trie, get_node_to_name_component_tuple_dict
from disjoint_set import DisjointSet

from definitions_to_runtime_terms_mappings import get_definitions_to_runtime_terms_mappings
from module_names_to_imported_names_to_runtime_objects import get_module_names_to_imported_names_to_runtime_objects

from collect_preliminary_typing_constraints import collect_and_resolve_typing_constraints
from relations import NonEquivalenceRelationGraph
from resolve_typing_constraints import resolve_typing_constraints
from runtime_term import RuntimeTerm
from trie import search
from typeshed_client_ex.client import Client

if __name__ == '__main__':
    directory_name = 'shell_sort'

    # Initialize Typeshed client
    client: Client = Client()

    (
        module_name_to_file_path_dict,
        module_name_to_module_node_dict,
        module_name_to_function_name_to_parameter_name_list_dict,
        module_name_to_class_name_to_method_name_to_parameter_name_list_dict,
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict
    ) = static_import_analysis.do_static_import_analysis(directory_name)

    sys.path.insert(0, directory_name)

    module_name_to_module = {
        module_name: importlib.import_module(module_name)
        for module_name, file_path in module_name_to_file_path_dict.items()
    }

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

    module_names_to_imported_names_to_runtime_objects = get_module_names_to_imported_names_to_runtime_objects(
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict,
        sys.modules
    )

    (
        nodes_to_attribute_counters,
        nodes_to_runtime_term_sets,
        nodes_providing_scope_to_parameter_lists,
        nodes_providing_scope_to_return_value_sets,
        nodes_providing_scope_to_yield_value_sets,
        nodes_providing_scope_to_send_value_sets,
        equivalence_relation_graph,
        other_relations_graph
    ) = collect_and_resolve_typing_constraints(
        top_level_class_definitions_to_runtime_classes,
        unwrapped_runtime_functions_to_named_function_definitions,
        module_names_to_imported_names_to_runtime_objects,
        module_name_to_module_node_dict
    )

    (
        equivalent_type_variables,
        type_variable_relations,
        type_variables_to_attribute_counters,
        equivalent_type_variable_attribute_counters,
        equivalent_type_variable_runtime_terms,
        equivalent_node_disjoint_set_top_nodes_non_equivalence_relation_graph
    ) = resolve_typing_constraints(
            unwrapped_runtime_functions_to_named_function_definitions,
            nodes_to_attribute_counters,
            nodes_to_runtime_term_sets,
            nodes_providing_scope_to_parameter_lists,
            nodes_providing_scope_to_return_value_sets,
            nodes_providing_scope_to_yield_value_sets,
            nodes_providing_scope_to_send_value_sets,
            equivalence_relation_graph,
            other_relations_graph,
            client
    )

    # equivalent_type_variables = DisjointSet()
    #
    # for first, second in equivalence_relation_graph.edges:
    #     equivalent_type_variables.union(first, second)

    representative_type_variable_to_equivalent_set: defaultdict[ast.AST, set[ast.AST]] = defaultdict(set)
    for representative_type_variable, equivalent_set in equivalent_type_variables.itersets():
        representative_type_variable_to_equivalent_set[representative_type_variable] = equivalent_set

    ast_node_namespace_trie_root = build_ast_node_namespace_trie(
        module_name_list,
        module_node_list
    )

    node_to_name_component_tuple_dict = get_node_to_name_component_tuple_dict(
        ast_node_namespace_trie_root
    )


    def get_indentation(indent_level: int = 0) -> str:
        return '    ' * indent_level


    def print_equivalent_sets(
            equivalent_sets: typing.Iterable[set[ast.AST]],
            representative_elements: typing.Iterable[ast.AST],
            type_variable_relations: NonEquivalenceRelationGraph,
            equivalent_type_variable_attribute_counters: defaultdict[ast.AST, AttributeCounter],
            equivalent_type_variable_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]],
            indent_level: int = 0
    ):
        for i, (equivalent_set, representative_element) in enumerate(
                zip(
                    equivalent_sets,
                    representative_elements
                )
        ):
            print(get_indentation(indent_level), f'Equivalent set {i}:')
            print_equivalent_set(
                equivalent_set,
                representative_element,
                type_variable_relations,
                equivalent_type_variable_attribute_counters,
                equivalent_type_variable_runtime_term_sets,
                indent_level + 1
            )


    def print_equivalent_set(
            equivalent_set: set[ast.AST],
            representative_element: ast.AST,
            type_variable_relations: NonEquivalenceRelationGraph,
            equivalent_type_variable_attribute_counters: defaultdict[ast.AST, AttributeCounter],
            equivalent_type_variable_runtime_term_sets: defaultdict[ast.AST, set[RuntimeTerm]],
            indent_level: int = 0
    ):
        print(get_indentation(indent_level), 'Nodes:')
        for node in equivalent_set:
            print_node(
                node,
                type_variable_relations,
                indent_level + 1
            )

        print(get_indentation(indent_level), 'Collective attribute counter:')
        print_attribute_counter(
            equivalent_type_variable_attribute_counters[representative_element],
            indent_level + 1
        )

        print(get_indentation(indent_level), 'Collective runtime terms:')
        print_runtime_term_set(
            equivalent_type_variable_runtime_term_sets[representative_element],
            indent_level + 1
        )


    def print_node(
            node: ast.AST,
            type_variable_relations: NonEquivalenceRelationGraph,
            indent_level: int = 0,
            exclude: frozenset[ast.AST] = frozenset()
    ):
        if node in exclude:
            return

        print(get_indentation(indent_level), node)
        print(get_indentation(indent_level + 1), 'Unparsed representation:', ast.unparse(node))
        print(get_indentation(indent_level + 1), 'Line number:', getattr(node, 'lineno', None))
        print(get_indentation(indent_level + 1), 'Related nodes:')
        print_related_nodes(node, type_variable_relations, indent_level + 2, exclude | frozenset([node]))


    def print_attribute_counter(
            attribute_counter: AttributeCounter,
            indent_level: int = 0
    ):
        for key, value in attribute_counter.items():
            print(get_indentation(indent_level), key, value)


    def print_runtime_term_set(
            runtime_term_set: set[RuntimeTerm],
            indent_level: int = 0
    ):
        for runtime_term in runtime_term_set:
            print(get_indentation(indent_level), runtime_term)


    def print_related_nodes(
            node: ast.AST,
            type_variable_relations: NonEquivalenceRelationGraph,
            indent_level: int = 0,
            exclude: frozenset[ast.AST] = frozenset()
    ):
        for relation_tuple, related_node_set in type_variable_relations.get_out_edges_by_relation_tuple(node).items():
            print(get_indentation(indent_level), relation_tuple)
            for related_node in related_node_set:
                print_node(related_node, type_variable_relations, indent_level + 1, exclude)

    while True:
        user_input = input('Enter the full containing namespace and name of the node (e.g. module_name class_name function_name name):')
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

        representative_type_variable_list = list({
            equivalent_type_variables.find(node)
            for node in node_set
        })

        equivalent_set_list = [
            representative_type_variable_to_equivalent_set[representative_type_variable]
            for representative_type_variable in representative_type_variable_list
        ]

        print_equivalent_sets(
            equivalent_set_list,
            representative_type_variable_list,
            type_variable_relations,
            equivalent_type_variable_attribute_counters,
            equivalent_type_variable_runtime_terms
        )
