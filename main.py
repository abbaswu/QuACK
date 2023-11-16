import argparse
import json
import logging
import os.path

from query_result_dict import QueryDict, ResultDict, generate_query_dict, raw_result_dict_from_result_dict
from run_mypy_and_parse_output import run_mypy_and_parse_output
from run_type_inference_method_and_postprocess_results import run_type_inference_method_and_postprocess_results
from static_import_analysis import do_static_import_analysis
from type_weaving import weave_types_for_project


if __name__ == '__main__':
    # Set up logging
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, required=True)
    parser.add_argument('-s', '--module-search-path', type=str, required=True)
    parser.add_argument('-p', '--module-prefix', type=str, required=True)
    parser.add_argument('-b', '--before-type-weaving-mypy-output-dataframe', type=str, required=True)
    parser.add_argument('-a', '--after-type-weaving-mypy-output-dataframe', type=str, required=True)
    parser.add_argument('-o', '--output-json', type=str, required=True)
    parser.add_argument('-t', '--time-output-json', type=str, required=True)
    parser.add_argument('-r', '--raw-output-directory', type=str, required=True)
    args = parser.parse_args()

    method: str = args.method
    module_search_path: str = args.module_search_path
    module_prefix: str = args.module_prefix
    before_type_weaving_mypy_output_dataframe: str = args.before_type_weaving_mypy_output_dataframe
    after_type_weaving_mypy_output_dataframe: str = args.after_type_weaving_mypy_output_dataframe
    output_json: str = args.output_json
    time_output_json: str = args.time_output_json
    raw_output_directory: str = args.raw_output_directory

    # Do static import analysis
    (
        module_name_to_file_path_dict,
        module_name_to_function_name_to_parameter_name_list_dict,
        module_name_to_class_name_to_method_name_to_parameter_name_list_dict,
        module_name_to_import_tuple_set_dict,
        module_name_to_import_from_tuple_set_dict
    ) = do_static_import_analysis(module_search_path, module_prefix)
    
    module_list: list[str] = list(module_name_to_file_path_dict.keys())

    # Generate query dict
    query_dict: QueryDict = generate_query_dict(
        module_name_to_file_path_dict,
        module_name_to_function_name_to_parameter_name_list_dict,
        module_name_to_class_name_to_method_name_to_parameter_name_list_dict
    )

    # Run type inference method and postprocess results
    result_dict, time_output = run_type_inference_method_and_postprocess_results(
        method,
        query_dict,
        module_search_path,
        module_prefix,
        module_name_to_class_name_to_method_name_to_parameter_name_list_dict,
        module_name_to_import_from_tuple_set_dict,
        raw_output_directory
    )
    
    # Run mypy and parse output before type weaving
    mypy_output_dataframe_before_type_weaving = run_mypy_and_parse_output(
        module_search_path,
        module_list
    )

    # Run type weaving
    weave_types_for_project(
        module_name_to_file_path_dict,
        result_dict
    )

    # Run mypy and parse output after type weaving
    mypy_output_dataframe_after_type_weaving = run_mypy_and_parse_output(
        module_search_path,
        module_list
    )

    with open(
        output_json,
        'w'
    ) as fp:
        json.dump(raw_result_dict_from_result_dict(result_dict), fp, indent=4)
    
    with open(
        time_output_json,
        'w'
    ) as fp:
        json.dump(time_output, fp, indent=4)
    
    mypy_output_dataframe_before_type_weaving.to_csv(before_type_weaving_mypy_output_dataframe)
    
    mypy_output_dataframe_after_type_weaving.to_csv(after_type_weaving_mypy_output_dataframe)

