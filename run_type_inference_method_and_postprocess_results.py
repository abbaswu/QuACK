import json
import os.path
import subprocess
import tempfile

from typing import Callable

import parse_stray_type_annotations
import parse_hityper_type_annotations

from query_result_dict import QueryDict, RawResultDict, ResultDict, result_dict_from_raw_result_dict
from type_inference_result import TypeInferenceResult, parse


def run_type_inference_method_and_postprocess_results(
    method: str,
    query_dict: QueryDict,
    module_search_path: str,
    module_prefix: str,
    module_name_to_class_name_to_method_name_to_parameter_name_list_dict: dict[str, dict[str, dict[str, list[str]]]],
    module_name_to_import_from_tuple_set_dict: dict[str, set[tuple[str, str, str]]],
    raw_output_directory: str
):
    # Create a temporary file
    with tempfile.TemporaryDirectory() as temporary_directory_name:
        it = tempfile._get_candidate_names()

        query_dict_temporary_file_name = next(it)
        query_dict_temporary_file_path = os.path.join(temporary_directory_name, query_dict_temporary_file_name)
        with open(query_dict_temporary_file_path, 'w') as fp:
            json.dump(query_dict, fp, indent=4)

        output_temporary_file_name = next(it)
        output_temporary_file_path = os.path.join(temporary_directory_name, output_temporary_file_name)

        time_temporary_file_name = next(it)
        time_temporary_file_path = os.path.join(temporary_directory_name, time_temporary_file_name)
        
        result = subprocess.run(
            [
                'bash',
                'run_type_inference_method.sh',
                '-m',
                method,
                '-q',
                query_dict_temporary_file_path,
                '-s',
                module_search_path,
                '-o',
                output_temporary_file_path,
                '-p',
                module_prefix,
                '-t',
                time_temporary_file_path,
                '-r',
                raw_output_directory
            ]
        )

        # If the script returned a non-zero exit code, raise an exception
        if result.returncode != 0:
            raise Exception(f"Script returned non-zero exit code {result.returncode}.")

        # Load the JSON
        # May raise an exception
        with open(output_temporary_file_path, 'r') as fp:
            raw_result_dict: RawResultDict = json.load(fp)

        # Parse raw result dict
        method_to_type_annotation_parser_dict: dict[
            str,
            Callable[[str, str], TypeInferenceResult]
        ] = {
            'stray': parse_stray_type_annotations.get_type_annotation_parser(
                module_name_to_class_name_to_method_name_to_parameter_name_list_dict,
                module_name_to_import_from_tuple_set_dict
            ),
            'hityper': parse_hityper_type_annotations.get_type_annotation_parser(
                module_name_to_class_name_to_method_name_to_parameter_name_list_dict,
                module_name_to_import_from_tuple_set_dict
            )
        }

        with open(time_temporary_file_path, 'r') as fp:
            time_output = json.load(fp)

        return result_dict_from_raw_result_dict(
            raw_result_dict,
            method_to_type_annotation_parser_dict.get(
                method,
                lambda module_name, type_annotation_string: parse(type_annotation_string)
            )
        ), time_output
