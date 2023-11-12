#!/bin/bash

# Functions

verify_module_list () {
module_list="$1"

python3 -u -c """
import json

module_list_string='''$module_list'''
module_list = json.loads(module_list_string)

assert isinstance(module_list, list)
for module_name in module_list:
    assert isinstance(module_name, str)
"""
}


build_mypy_module_option_string () {
module_list="$1"

python3 -u -c """
import json

module_list_string='''$module_list'''
module_list = json.loads(module_list_string)

mypy_module_option_string_components = []
for module in module_list:
    mypy_module_option_string_components.append('-m')
    mypy_module_option_string_components.append(repr(module))

print(' '.join(mypy_module_option_string_components))
"""
}


# Variables from command-line arguments

# Module search path, pass with option `-s`
module_search_path=

# Module list, pass with option `-l`
module_list=

while getopts ':l:s:' name
do
    case $name in
        l)
            module_list="$OPTARG"
            ;;
        s)
            # Module search path MUST NOT end with '/'s, otherwise breaks hityper
            module_search_path="$(echo "$OPTARG" | sed 's/\/\+$//')"
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

if [ -z "$module_list" ] || [ -z "$module_search_path" ]
then
    echo "Usage: $0 -l <module_list> -s <module_search_path>" >&2
    exit 1
fi

if ! verify_module_list "$module_list"
then
    echo "Invalid module list provided!" >&2
    echo "A valid query dict should be a valid JSON list containing strings." >&2
    exit 1
fi

if ! [ -d "$module_search_path" ]
then
    echo "Module search path `$module_search_path` is not a directory!" >&2
    exit 1
fi

# Install from requirements.txt
if [ -f "${module_search_path}/requirements.txt" ]
then
    conda run --no-capture-output --name mypy pip install -r "${module_search_path}/requirements.txt" 1>&2
fi

current_working_directory="$(pwd)"

# Enter module search path
cd "$module_search_path"

# Build mypy module option string
# Before (module_list): '["1", "2", "3"]'
# After (module_option_string): "-m '1' -m '2' -m '3'"
# Credits: https://chat.openai.com/
mypy_module_option_string="$(build_mypy_module_option_string "$module_list")"

# Run mypy
mypy_command="conda run --no-capture-output --name mypy mypy $mypy_module_option_string"

eval "$mypy_command"

# Exit module search path
cd "$current_working_directory"
