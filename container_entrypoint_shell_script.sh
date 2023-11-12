#!/bin/bash

set -e
set -o pipefail

# Constants

MOUNTED_MODULE_SEARCH_PATH='/mnt/mounted_module_search_path'

OUTPUT_PATH='/mnt/output_path'

# MUST NOT end with a '/', otherwise breaks hityper
LOCAL_MODULE_SEARCH_PATH='/tmp/local_module_search_path'

# Functions

is_port_in_use () {
port="$1"

python3 -u -c """
import socket
import sys


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


if is_port_in_use($port):
    sys.exit(0)
else:
    sys.exit(1)
"""
}


# Variables from command-line arguments

# Method, pass with option `-m`
method=

# Module prefix, pass with `-p`
module_prefix=

while getopts ':m:p:' name
do
    case $name in
        m)
            method="$OPTARG"
            ;;
        p)
            module_prefix="$OPTARG"
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

if [ ! -d "$MOUNTED_MODULE_SEARCH_PATH" ]
then
    echo "Module search path is not mounted to ${MOUNTED_MODULE_SEARCH_PATH}" >&2
    echo "Please provide a mount point with your Docker run command: " >&2
    echo "docker run --net=host -v <module_search_path>:${MOUNTED_MODULE_SEARCH_PATH}:ro -v <output_path>:${OUTPUT_PATH} ..." >&2
    exit 1
fi

if [ ! -d "$OUTPUT_PATH" ]
then
    echo "Output path is not mounted to ${OUTPUT_PATH}" >&2
    echo "Please provide a mount point with your Docker run command: " >&2
    echo "docker run --net=host -v <module_search_path>:${MOUNTED_MODULE_SEARCH_PATH}:ro -v <output_path>:${OUTPUT_PATH} ..." >&2
    exit 1
fi

if [ -z "$method" ] || [ -z "$module_prefix" ]
then
    echo "Usage: $0 -m <method> -p <module_prefix>" >&2
    exit 1
fi

if ! is_port_in_use 5001
then
    echo "Type4Py server not available on port 5001!" >&2
    echo "Make sure that you have started Type4Py server outside of this container:" >&2
    echo '''
    docker pull ghcr.io/saltudelft/type4py:latest
    docker run -d -p 5001:5010 -it ghcr.io/saltudelft/type4py:latest
    ''' >&2
    echo "docker run --net=host -v <module_search_path>:${MOUNTED_MODULE_SEARCH_PATH}:ro -v <output_path>:${OUTPUT_PATH} ..." >&2
    exit 1
fi

# Preprocessing

# Copy contents from $MOUNTED_MODULE_SEARCH_PATH to $LOCAL_MODULE_SEARCH_PATH
cp -R "$MOUNTED_MODULE_SEARCH_PATH" "$LOCAL_MODULE_SEARCH_PATH"

# Strip type annotations from Python files in $LOCAL_MODULE_SEARCH_PATH
python3 /root/strip_type_annotations.py --directory "$LOCAL_MODULE_SEARCH_PATH"

# Change directory to /root
cd /root

# Run method, modifying the contents of $LOCAL_MODULE_SEARCH_PATH
python3 /root/main.py -m "$method" -s "$LOCAL_MODULE_SEARCH_PATH" -p "$module_prefix"
return_code="$?"

# Copy contents from $LOCAL_MODULE_SEARCH_PATH to $OUTPUT_PATH
if [ "$return_code" -eq 0 ]
then
    cp -R -v -f "$LOCAL_MODULE_SEARCH_PATH"/* "$OUTPUT_PATH"/
fi
