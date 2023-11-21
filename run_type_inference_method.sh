#!/bin/bash

set -e
set -o pipefail

# Constants

VERIFY_QUERY_DICT='/root/verify_query_dict.py'

# Variables from command-line arguments

# Method, pass with option `-m`
method= 

# Query dict, pass with option `-q`
query_dict=

# Module search path, pass with option `-s`
module_search_path=

# Output file path, pass with option `-o`
output_file_path=

# Module prefix, pass with option `-p`
module_prefix=

# Time output file path, pass with option `-t`
time_output_file_path=

# Raw output directory, pass with option `-r`
raw_output_directory=

while getopts ':m:q:s:o:p:t:r:' name
do
    case $name in
        m)
            method="$OPTARG"
            ;;
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
        p)
            module_prefix="$OPTARG"
            ;;
        t)
            time_output_file_path="$OPTARG"
            ;;
        r)
            raw_output_directory="$OPTARG"
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

if [ -z "$method" ] || [ -z "$query_dict" ] || [ -z "$module_search_path" ] || [ -z "$output_file_path" ] || [ -z "$module_prefix" ] || [ -z "$time_output_file_path" ] || [ -z "$raw_output_directory" ]
then
    echo "Usage: $0 -m <method> -q <query_dict> -s <module_search_path> -o <output_file_path> -p <module_prefix> -t <time_output_file_path> -r <raw_output_directory>" >&2
    exit 1
fi

if ! python3 "$VERIFY_QUERY_DICT" --query-dict "$query_dict"
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

# Run method

case "$method" in
    # pyre)
    #     pyre_output_file="$module_search_path/pyre_output_file"

    #     conda run --no-capture-output --name pyre pip install -r "${module_search_path}/requirements.txt" 1>&2
    #     # Prompt:
    #     # Also initialize watchman in the current directory? [Y/n]
    #     # Which directory(ies) should pyre analyze? (Default: `.`):
    #     printf "Y\n${module_search_path}\n" | conda run --no-capture-output --name pyre pyre init
    #     conda run --no-capture-output --name pyre pyre infer --print-only > "$pyre_output_file" 2>&1

    #     cat "$pyre_output_file" | python3 "$(pwd)/parse_pyre_stdout.py" --query-dict "$query_dict" > "$output_file_path"
    #     ;;
    # pytype)
    #     conda run --no-capture-output --name pytype pip install -r "${module_search_path}/requirements.txt" 1>&2
    #     conda run --no-capture-output --name pytype pytype --keep-going "$module_search_path" 1>&2

    #     pytype_pyi_directory="$(pwd)/.pytype/pyi"
        
    #     # Import-based analysis, MUST use pytype's virtual environment that has all the requirements installed
    #     conda run --no-capture-output --name pytype python3 "$(pwd)/parse_pytype_result_directory.py" --query-dict "$query_dict" --module-search-path "$module_search_path" --pytype-pyi-directory "$pytype_pyi_directory" > "$output_file_path"
    #     ;;
    quack)
        output_file="$module_search_path/output_file.json"
        
        if [ -f "${module_search_path}/requirements.txt" ]; then
            conda run --no-capture-output --name quack pip install -r "${module_search_path}/requirements.txt" 1>&2
        fi

        PYTHONPATH="$module_search_path" /usr/bin/time -f '{"maximum resident set size in KB": %M, "elapsed real time (wall clock) in seconds": %e}' -o "$time_output_file_path" conda run --no-capture-output --name quack python3 "$(pwd)/quack/main.py" --module-search-path "$module_search_path" --module-prefix "$module_prefix" --output-file "$output_file_path"
        ;;
    quack-no-induced-equivalent-relation-resolution)
        output_file="$module_search_path/output_file.json"
        
        if [ -f "${module_search_path}/requirements.txt" ]; then
            conda run --no-capture-output --name quack pip install -r "${module_search_path}/requirements.txt" 1>&2
        fi

        PYTHONPATH="$module_search_path" /usr/bin/time -f '{"maximum resident set size in KB": %M, "elapsed real time (wall clock) in seconds": %e}' -o "$time_output_file_path" conda run --no-capture-output --name quack python3 "$(pwd)/quack/main.py" --no-induced-equivalent-relation-resolution --module-search-path "$module_search_path" --module-prefix "$module_prefix" --output-file "$output_file_path"
        ;;
    quack-no-attribute-access-propagation)
        output_file="$module_search_path/output_file.json"
        
        if [ -f "${module_search_path}/requirements.txt" ]; then
            conda run --no-capture-output --name quack pip install -r "${module_search_path}/requirements.txt" 1>&2
        fi

        PYTHONPATH="$module_search_path" /usr/bin/time -f '{"maximum resident set size in KB": %M, "elapsed real time (wall clock) in seconds": %e}' -o "$time_output_file_path" conda run --no-capture-output --name quack python3 "$(pwd)/quack/main.py" --no-attribute-access-propagation --module-search-path "$module_search_path" --module-prefix "$module_prefix" --output-file "$output_file_path"
        ;;
    quack-no-stdlib-function-call-propagation)
        output_file="$module_search_path/output_file.json"
        
        if [ -f "${module_search_path}/requirements.txt" ]; then
            conda run --no-capture-output --name quack pip install -r "${module_search_path}/requirements.txt" 1>&2
        fi

        PYTHONPATH="$module_search_path" /usr/bin/time -f '{"maximum resident set size in KB": %M, "elapsed real time (wall clock) in seconds": %e}' -o "$time_output_file_path" conda run --no-capture-output --name quack python3 "$(pwd)/quack/main.py" --no-stdlib-function-call-propagation --module-search-path "$module_search_path" --module-prefix "$module_prefix" --output-file "$output_file_path"
        ;;
    quack-no-user-defined-function-call-propagation)
        output_file="$module_search_path/output_file.json"
        
        if [ -f "${module_search_path}/requirements.txt" ]; then
            conda run --no-capture-output --name quack pip install -r "${module_search_path}/requirements.txt" 1>&2
        fi

        PYTHONPATH="$module_search_path" /usr/bin/time -f '{"maximum resident set size in KB": %M, "elapsed real time (wall clock) in seconds": %e}' -o "$time_output_file_path" conda run --no-capture-output --name quack python3 "$(pwd)/quack/main.py" --no-user-defined-function-call-propagation --module-search-path "$module_search_path" --module-prefix "$module_prefix" --output-file "$output_file_path"
        ;;
    stray)
        if [ -f "${module_search_path}/requirements.txt" ]; then
            conda run --no-capture-output --name stray pip install -r "${module_search_path}/requirements.txt" 1>&2
        fi

        /usr/bin/time -f '{"maximum resident set size in KB": %M, "elapsed real time (wall clock) in seconds": %e}' -o "$time_output_file_path" /bin/bash run_stray.sh -q "$query_dict" -s "$module_search_path" -o "$output_file_path" -r "$raw_output_directory"
        ;;
    hityper)
        if [ -f "${module_search_path}/requirements.txt" ]; then
            conda run --no-capture-output --name hityper pip install -r "${module_search_path}/requirements.txt" 1>&2
        fi

        /usr/bin/time -f '{"maximum resident set size in KB": %M, "elapsed real time (wall clock) in seconds": %e}' -o "$time_output_file_path" /bin/bash run_hityper.sh -q "$query_dict" -s "$module_search_path" -o "$output_file_path" -r "$raw_output_directory"
        ;;
    *)
        # error
        echo "Invalid method `$method`." >&2
        exit 1
esac
