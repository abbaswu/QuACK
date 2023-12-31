# QuACK

All source code for "QuACK: Attribute-centric Type Inference for Python" in the form of a Docker container. **Please read this file carefully before playing around.**

Note: Our proposed attribute-centric type inference method is currently Work in Progress. Only the Docker container infrastructure, which includes code for our evaluation pipeline:

- Stripping existing type annotations from a copy of a given Python project
- Analyzing the Python project
- Running the baseline type inference methods [Stray](https://github.com/ksun212/Stray) and [HiTyper](https://github.com/JohnnyPeng18/HiTyper) on the Python project
- Post-processing their results
- Weaving their results into the Python project
- Running mypy on all modules within the Python project
- Recording mypy's error messages

is currently included within the repository.

## Requirements and Assumptions

- You are on a Unix-like system running on an x86-64 machine with Docker installed.
- Your Python project under analysis has all modules written in **pure Python** (no Cython/C code, no FFIs).
- You should know the `absolute path of the directory containing Python modules`. **From that path, the Python interpreter should be able to successfully import every module within the Python project.** This depends from project to project. For example:
    - If you have cloned the [NetworkX](https://github.com/networkx/networkx.git) repository to `/tmp/networkx`, that directory should be `/tmp/networkx`.
    - If you have cloned the [typing_extensions](https://github.com/python/typing_extensions) repository to `/tmp/typing_extensions`, that directory should be `/tmp/typing_extensions/`
- Your Python project under analysis should either **have no dependencies** or **have all dependencies explicitly listed in a `requirements.txt` file under the `absolute path of the directory containing Python modules`**.

## Instructions

Build the container from the GitHub repository.

```bash
docker build --tag quack .
```

Start Type4Py server (required for the baseline type inference method [HiTyper](https://github.com/JohnnyPeng18/HiTyper)).

```bash
docker pull ghcr.io/saltudelft/type4py:latest
docker run -d -p 5001:5010 -it ghcr.io/saltudelft/type4py:latest
```

Use the Docker container to run a specified `type inference method` on a Python project.

- Provide the `absolute path of the directory containing Python modules`.
- Provide the `absolute path of the output directory`. The following intermediate and final results will be saved in the output directory:
    - A type-weaved copy of the original Python project
    - A JSON file `result.json` storing the (normalized) results of running the specified `type inference method`.
    - Two CSV files, `mypy_output_dataframe_before_type_weaving.csv` and `mypy_output_dataframe_after_type_weaving.csv`, storing error messages generated by mypy before and after type weaving.
- Provide a `type inference method` (`'stray'`, `'hityper'`, etc.).
- Provide a `module prefix`. This filters out irrelevant modules that shouldn't be analyzed, such as installation files, example files, or test files. For example, given the [NetworkX](https://github.com/networkx/networkx.git) repository, providing the `module prefix` of `networkx` allows us to analyze files such as `networkx/algorithms/clique.py` while skipping files such as `examples/external/plot_igraph.py`.

```bash
docker run \
--rm \
--net=host \
-v <absolute path of the directory containing Python modules>:/mnt/mounted_module_search_path:ro \
-v <absolute path of the output directory>:/mnt/output_path \
quack \
-m <type inference method> \
-p <module prefix>
```

## Code Organization

- `Dockerfile`: Self-explanatory.
- `Miniconda3-latest-Linux-x86_64.sh`: Used for installing a Conda environment within the Docker container.
- `Stray/`, `HiTyper/`: Source code of the baseline type inference methods [Stray](https://github.com/ksun212/Stray) and [HiTyper](https://github.com/JohnnyPeng18/HiTyper).
- `container_entrypoint_shell_script.sh`: Entrypoint Shell script of the Docker container. Copies `absolute path of the directory containing Python modules`, strips existing type annotations, installs dependencies, and invokes:
    - `main.py`: Orchestrates the evaluation pipeline after that. Invokes:
        - `run_type_inference_method_and_postprocess_results.py`, `run_type_inference_method.sh`: Runs the specified `type inference method` and postprocess results. *Modify these files to support more type inference methods within the evaluation pipeline.*
        - `type_weaving.py`: Self-explanatory.
        - `run_mypy_and_parse_output.py`: Self-explanatory.
