# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

if (NOT EXISTS ${Python3_SITELIB}/skbuild)
  # In case this is messed up by `/usr/local/python/site-packages` vs `/usr/python/site-packages`, check pip itself.
  execute_process(
      COMMAND bash "-c" "pip show scikit-build | sed -n -e 's/Location: //p'"
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (NOT EXISTS ${PYTHON_SITE_PACKAGES}/skbuild)
    message(SEND_ERROR "Scikit-build is not installed. CMake may not be able to find Cython. Install scikit-build with `pip install scikit-build`")
  else()
    list(APPEND CMAKE_MODULE_PATH "${PYTHON_SITE_PACKAGES}/skbuild/resources/cmake")
  endif()
else ()
  list(APPEND CMAKE_MODULE_PATH "${Python3_SITELIB}/skbuild/resources/cmake")
endif()