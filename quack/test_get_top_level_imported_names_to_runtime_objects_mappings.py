"""
To generate an HTML coverage report:

- Run Coverage: coverage run --source=get_top_level_imported_names_to_runtime_objects_mappings test_get_top_level_imported_names_to_runtime_objects_mappings.py
- Generate an HTML report: coverage html
"""

import importlib
import sys

import static_import_analysis

from module_names_to_imported_names_to_runtime_objects_singleton import get_module_names_to_imported_names_to_runtime_objects


if __name__ == '__main__':
    (
        module_name_to_file_path_dict,
        module_name_to_module_node_dict,
        module_name_to_function_name_to_parameter_name_list_dict,
        module_name_to_class_name_to_method_name_to_parameter_name_list_dict,
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict
    ) = static_import_analysis.do_static_import_analysis('test_project')

    sys.path.insert(0, 'test_project')

    module_name_to_module = {
        module_name: importlib.import_module(module_name)
        for module_name, file_path in module_name_to_file_path_dict.items()
    }

    module_names_to_imported_names_to_runtime_objects = get_module_names_to_imported_names_to_runtime_objects(
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict,
        module_name_to_module
    )

    assert set(module_names_to_imported_names_to_runtime_objects['data'].keys()) == {'aiohttp', 'process_numbers', 'sqrt'}
    assert set(module_names_to_imported_names_to_runtime_objects['main'].keys()) == {'asyncio', 'Outer', 'public_function', 'PublicClass'}
    assert set(module_names_to_imported_names_to_runtime_objects['nested_functions_and_classes'].keys()) == set()
    assert set(module_names_to_imported_names_to_runtime_objects['utils'].keys()) == {'sqrt'}
