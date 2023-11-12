# Constants

STRAY_ROOT='/root/stray'

PRINT_STRAY_RESULT_DIRECTORY='/root/parse_stray_result_directory.py'

# Functions

verify_query_dict () {
query_dict="$1"

python3 -u -c """
import json

query_dict_string='''$query_dict'''
query_dict = json.loads(query_dict_string)

assert isinstance(query_dict, dict)
for module_name, module_name_level_query_dict in query_dict.items():
    assert isinstance(module_name, str)
    assert isinstance(module_name_level_query_dict, dict)
    for class_name_or_global, class_name_or_global_level_query_dict in module_name_level_query_dict.items():
        assert isinstance(class_name_or_global, str)
        assert isinstance(class_name_or_global_level_query_dict, dict)
        for function_name, parameter_name_or_return_level_query_dict in class_name_or_global_level_query_dict.items():
            assert isinstance(function_name, str)
            for parameter_name_or_return in parameter_name_or_return_level_query_dict:
                assert isinstance(parameter_name_or_return, str)
"""
}

# Variables from command-line arguments

# Query dict, pass with option `-q`
query_dict=

# Module search path, pass with option `-s`
module_search_path=

# Output file path, pass with option `-o`
output_file_path=

while getopts ':q:s:o:' name
do
    case $name in
        q)
            query_dict="$OPTARG"
            ;;
        s)
            # Module search path MUST NOT end with '/'s, otherwise breaks hityper
            module_search_path="$(echo "$OPTARG" | sed 's/\/\+$//')"
            ;;
        o)
            output_file_path="$OPTARG"
            ;;
        :)
            echo "Option -$OPTARG requires an argument"
            ;;
        ?)
            echo "Invalid option -$OPTARG"
            ;;
    esac
done

# Sanity check

if [ -z "$query_dict" ] || [ -z "$module_search_path" ] || [ -z "$output_file_path" ]
then
    echo "Usage: $0 -q <query_dict> -s <module_search_path> -o <output_file_path>" >&2
    exit 1
fi

if ! verify_query_dict "$query_dict"
then
    echo "Invalid query dict provided!" >&2
    echo "A valid query dict should be a valid JSON string that can be represented using the following Python data type:" >&2
    echo '''
    dict[
        str, # module_name
        dict[
            str, # class_name_or_global
            dict[
                str, # function_name
                typing.Iterable[
                    str # parameter_name_or_return
                ]
            ]
        ]
    ''' >&2
    exit 1
fi

if ! [ -d "$module_search_path" ]
then
    echo "Module search path `$module_search_path` is not a directory!" >&2
    exit 1
fi

mkdir -p "$STRAY_ROOT/result"
mkdir -p "$STRAY_ROOT/results"

absolute_module_search_path="$(realpath "$module_search_path")"

current_working_directory="$(pwd)"

cd "$STRAY_ROOT"

set +e
python3 "$current_working_directory/print_python_file_paths.py" -q "$query_dict" -s "$absolute_module_search_path" | while read python_file_path
do
    directory_name="$(dirname "$python_file_path")"
    file_name="$(basename "$python_file_path")"
    file_name_without_extension="${file_name%.py}"

    echo conda run --no-capture-output --name stray python3 -m predict "$directory_name" check "$file_name_without_extension" 1>&2
    conda run --no-capture-output --name stray python3 -m predict "$directory_name" check "$file_name_without_extension" 1>&2
    echo conda run --no-capture-output --name stray python3 -m predict "$directory_name" check "$file_name_without_extension" 1>&2
    conda run --no-capture-output --name stray python3 -m predict "$directory_name" predict "$file_name_without_extension" 1>&2
done
set -e

cd "$current_working_directory"

python3 "$PRINT_STRAY_RESULT_DIRECTORY" --stray-result-directory "$STRAY_ROOT/result" --query-dict "$query_dict" --absolute-module-search-path "$absolute_module_search_path" --output-file "$output_file_path"
