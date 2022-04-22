**General architectural ideas**

We build three libraries:
- libmorpheus : defines all the python aware library code for morpheus, and interface proxies for python modules.
  - Interface proxies are designed to provide a single consolidated point of interaction between the morpheus 
  library code and their associated pybind11 module definitions.
  - Please avoid declaring adhoc functions/interfaces that link to python modules.
- libmorpheus_utils : matx and table manipulation functions.
- libcudf_helpers : small bridge module used to extract cython based dataframe, and series information from cuDF.


Python modules should be defined in `_lib/src/python_modules`, with an associated cmake declaration in 
`_lib/cmake/<module_name>.cmake` which can be included in `_lib/CMakeLists.txt`.