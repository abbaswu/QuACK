from query_result_dict import QueryDict, ResultDict, generate_query_dict, raw_result_dict_from_result_dict
from read_result_dict_from_output_json_file import *
from static_import_analysis import *
from strip_type_annotations import *
from use_mypy_to_check_each_type_prediction import *

import ast
import os
import os.path
import shutil
import tempfile

import pudb
from tqdm import tqdm

# Constants

METHODS: list[str] = [
    'stray',
    'hityper',
    'quack_new',
    # 'extract_type_annotations',
    # 'quack--parameters-only',
    # 'quack--return-values-only',
    # 'quack--no-induced-equivalent-relation-resolution',
    # 'quack--no-attribute-access-propagation',
    # 'quack--no-stdlib-function-call-propagation',
    # 'quack--no-user-defined-function-call-propagation',
    # 'quack--no-shortcut-single-class-covering-all-attributes',
    # 'quack--no-parameter-default-value-handling',
]
# DIRECTORY_CONTAINING_REPOSITORIES: str = '../experiments/top_python_packages_with_type_annotations'
DIRECTORY_CONTAINING_REPOSITORIES: str = '../experiments/top_python_packages'

# (repository_name, relative_module_search_path, module_prefix)

REPOSITORIES: list[tuple[str, str, str]] = [
    ('requests-2.31.0', '.', 'requests'),
    ('Pygments-2.15.1', '.', 'pygments'),
    ('networkx-3.1', '.', 'networkx'),
    ('boto3-1.28.10', '.', 'boto3'),
    ('gunicorn-21.2.0', '.', 'gunicorn'),
    ('tqdm-4.65.0', '.', 'tqdm'),
    ('python-dateutil-2.8.2', '.', 'dateutil'),
    ('pytz-2023.3', '.', 'pytz'),
    ('six-1.16.0', '.', 'six'),
    ('pytest-cov-4.1.0', 'src', 'pytest_cov'),
    ('notebook-7.0.0', '.', 'notebook'),
    ('peewee-3.16.2', '.', 'peewee'),
    ('seaborn-0.12.2', '.', 'seaborn'),
    # ('click-8.1.6', 'src', 'click'),
    # ('flake8-4.0.1', 'src', 'flake8'),
    # ('Flask-2.3.2', 'src', 'flask'),
    # ('ipython-8.14.0', '.', 'IPython'),
    # ('Jinja2-3.1.2', 'src', 'jinja2'),
    # ('pre_commit-3.3.3', '.', 'pre_commit'),
    # ('pylint-2.17.4', '.', 'pylint'),
    # ('sphinx-7.1.0', '.', 'sphinx'),
    # ('urllib3-2.0.4', 'src', 'urllib3'),
    # ('Werkzeug-2.3.6', 'src', 'werkzeug')

]

RESULT_DIRECTORY: str = '../experiments/result_directory'

RUN_MYPY_PREFIX = ['conda', 'run', '--no-capture-output', '--name', 'mypy']
RUN_QUACK_PREFIX = ['conda', 'run', '--no-capture-output', '--name', 'quack']

def run_quack(
    module_search_path,
    module_prefix,
    output_file,
    run_command_in_environment_prefix: list[str]
):
    # Install from requirements.txt
    requirements_txt_path = os.path.join(module_search_path, 'requirements.txt')
    if os.path.isfile(requirements_txt_path):
        result = subprocess.run(
            run_command_in_environment_prefix + [
                'pip',
                'install',
                '-r',
                requirements_txt_path
            ]
        )

        # If the script returned a non-zero exit code, raise an exception
        if result.returncode != 0:
            raise Exception(f"Script returned non-zero exit code {result.returncode}.")

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

    # Create a temporary file
    with tempfile.TemporaryDirectory() as temporary_directory_name:
        it = tempfile._get_candidate_names()

        query_dict_temporary_file_name = next(it)
        query_dict_temporary_file_path = os.path.join(temporary_directory_name, query_dict_temporary_file_name)
        with open(query_dict_temporary_file_path, 'w') as fp:
            json.dump(query_dict, fp, indent=4)

        # Run QuACK
        run_quack_result = subprocess.run(
            run_command_in_environment_prefix + [
                'python',
                os.path.join('quack_new', 'main.py'),
                '--query-dict',
                query_dict_temporary_file_path,
                '--module-search-path',
                module_search_path,
                '--module-prefix',
                module_prefix,
                '--output-file',
                output_file
            ]
        )

        # If the script returned a non-zero exit code, raise an exception
        if run_quack_result.returncode != 0:
            raise Exception(f"Script returned non-zero exit code {run_quack_result.returncode}.")


def copy_content_of_directory_to_directory(
    source_dir,
    target_dir
):
    # Copy each file and sub-directory from source_dir to type_prediction_temp_dir
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        destination_item = os.path.join(target_dir, item)

        if os.path.isdir(source_item):
            # If it's a directory, copy it recursively
            shutil.copytree(source_item, destination_item)
        else:
            # If it's a file, copy it
            shutil.copy2(source_item, destination_item)


def f(repository_name, relative_module_search_path, module_prefix):
    absolute_module_search_path = os.path.abspath(os.path.join(DIRECTORY_CONTAINING_REPOSITORIES, repository_name, relative_module_search_path))

    if not os.path.isdir(absolute_module_search_path):
        logging.error('Absolute module search path %s for repository %s does not exist!', absolute_module_search_path, repository_name)
        sys.exit(1)

    absolute_module_result_directory = os.path.abspath(os.path.join(RESULT_DIRECTORY, repository_name))
    method = 'quack_new_type_parameters'
    absolute_output_path = os.path.abspath(os.path.join(absolute_module_result_directory, method))

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as type_prediction_temp_dir:
        copy_content_of_directory_to_directory(absolute_module_search_path, type_prediction_temp_dir)

        strip_type_annotations(type_prediction_temp_dir)

        output_json_file_path = os.path.join(type_prediction_temp_dir, 'output.json')

        # Run method here
        run_quack(
            type_prediction_temp_dir,
            module_prefix,
            output_json_file_path,
            RUN_QUACK_PREFIX
        )

        (
            module_name_to_module_path_dict,
            *_
        ) = do_static_import_analysis(type_prediction_temp_dir, module_prefix)

        module_name_to_module_node_without_type_annotation_dict = {}
        for module_name, module_path in module_name_to_module_path_dict.items():
            with open(module_path, 'r') as fp:
                module_node = ast.parse(fp.read())
                module_name_to_module_node_without_type_annotation_dict[module_name] = module_node

        result_dict = read_result_dict_from_output_json_file(output_json_file_path)

        r = use_mypy_to_check_each_type_prediction(
            type_prediction_temp_dir,
            module_name_to_module_node_without_type_annotation_dict,
            module_name_to_module_path_dict,
            result_dict,
            RUN_MYPY_PREFIX
        )

        # Create absolute_output_path
        os.makedirs(absolute_output_path, exist_ok=True)

        # Save output_json
        shutil.copy2(output_json_file_path, os.path.join(absolute_output_path, 'output.json'))

        # Save result_dict 
        r.to_csv(os.path.join(
            absolute_output_path,
            'use_mypy_to_check_each_type_prediction.csv'
        ), index=False)

if __name__ == '__main__':
    import sys
    i = int(sys.argv[1])
    f(*(REPOSITORIES[i]))

