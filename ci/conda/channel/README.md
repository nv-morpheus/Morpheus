Creates a local conda channel using docker-compose and nginx. Can be helpful when testing new conda packages

To Use:
1. Ensure `docker-compose` is installed
2. Set the location of the conda-bld folder to host as a conda channel to the variable `$CONDA_REPO_DIR`
   1. i.e. `export CONDA_REPO_DIR=$CONDA_PREFIX/conda-bld`
3. Launch docker-compose
   1. `docker-compose up -d`
4. Install conda packages using the local channel
   1. `conda install -c http://localhost:8080 <my_package>`
