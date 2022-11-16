## Build libmorpheus Documentation

### Install Doxygen

Install Doxygen using conda.

```bash
conda install -c conda-forge doxygen
```

### Build libmorpheus HTML Documentation
There are multiple ways to build the libmorpheus HTML documentation.

* Run the `doxygen` command from the `morpheus/_lib/doxygen` directory containing the `Doxyfile`.
```bash
cd morpheus/_lib/doxygen
doxygen
```

* The libmorpheus documentation can also be built using `./scripts/compile.h` from the cmake build directory

```bash
export TARGET="docs_morpheus"
./scripts/compile.sh
```
Note: Doxygen reads and processes all appropriate source files under the `morpheus/_lib/include/` directory and generates the output in the `morpheus/_lib/doxygen/html/` directory. You can load the local `morpheus/_lib/doxygen/html/index.html` file generated there into any web browser to view the result.