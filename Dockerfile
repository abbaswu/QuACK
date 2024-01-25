# docker build --tag run_methods_container .

# Set the base image to Ubuntu (22.04)
FROM ubuntu

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential git watchman \
    && apt-get clean

# Install miniconda
# Put conda in path so we can use conda init
COPY Miniconda3-latest-Linux-x86_64.sh /root
ENV CONDA_ROOT /root/miniconda3
ENV PATH=$CONDA_ROOT/bin:$PATH
RUN /bin/bash /root/Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_ROOT && conda init 

# Set up Conda environments
# We CANNOT use conda activate with the RUN command
# so we use conda run -n env <command> instruction which actually runs <command> inside the conda environment.
# https://kevalnagda.github.io/conda-docker-tutorial
RUN conda create --name pyre -y python=3.10 \
    && conda run --name pyre pip install pyre-check

RUN conda create --name pytype -y python=3.10 \
    && conda run --name pytype pip install pytype

COPY hityper /root/hityper
RUN conda create --name hityper -y python=3.9 \
    && conda run --name hityper pip install --verbose /root/hityper \
    && conda run --name hityper pip install --verbose torch --index-url https://download.pytorch.org/whl/cpu

COPY stray /root/stray
RUN conda create --name stray -y python=3.9 \
    && conda run --name stray pip install --verbose -r /root/stray/requirements.txt \
    && cp -a /root/stray/data/pyi/numpy/. $CONDA_ROOT/envs/stray/lib/python3.9/site-packages/numpy \
    && cp -a /root/stray/data/pyi/matplotlib-stubs/. $CONDA_ROOT/envs/stray/lib/python3.9/site-packages/matplotlib-stubs

RUN conda create --name mypy -y python=3.10 \
    && conda run --name mypy pip install --verbose mypy

RUN conda create --name quack -y python=3.10 \
    && conda run --name quack pip install --verbose networkx numpy ordered_set pudb scipy typeshed_client

# Install packages required in entrypoint and data processing scripts
RUN conda run --name base pip install lark ordered_set pudb pandas

# Install /usr/bin/time
RUN  apt-get update && apt-get install -y time graphviz libgraphviz-dev && apt-get clean

RUN conda create --name extract_type_annotations -y python=3.10 \
    && conda run --name extract_type_annotations pip install --verbose pudb lark

# Copy entrypoint and data processing scripts
COPY quack /root/quack/
COPY quack_new /root/quack_new/
COPY static_import_analysis /root/static_import_analysis/
COPY extract_type_annotations /root/extract_type_annotations/
COPY *.py *.sh /root/

# command executable and version
ENTRYPOINT ["/bin/bash", "/root/container_entrypoint_shell_script.sh"]
# ENTRYPOINT ["/bin/bash"]
