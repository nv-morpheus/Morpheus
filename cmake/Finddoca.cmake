# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

set(DOCA_LIB_DIR "/opt/mellanox/doca/lib/x86_64-linux-gnu")

find_library(DOCA_LIBRARY_DIR NAMES doca_gpunetio HINTS ${DOCA_LIB_DIR})
find_path(DOCA_INCLUDE_DIR doca_gpunetio.h HINTS /opt/mellanox/doca/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(doca DEFAULT_MSG DOCA_LIBRARY_DIR DOCA_INCLUDE_DIR)

if (DOCA_FOUND)

    list(APPEND DOCA_SHARED_LIBS
        doca_eth
        doca_flow
        doca_gpunetio
    )

    list(APPEND DOCA_STATIC_LIBS
        doca_gpunetio_device
    )

    foreach(DOCA_LIBRARY_NAME IN LISTS DOCA_SHARED_LIBS)
        find_library(DOCA_LIBRARY_${DOCA_LIBRARY_NAME} NAMES ${DOCA_LIBRARY_NAME} HINTS ${DOCA_LIB_DIR})
        add_library(${DOCA_LIBRARY_NAME} SHARED IMPORTED GLOBAL)
        add_library(doca::${DOCA_LIBRARY_NAME} ALIAS ${DOCA_LIBRARY_NAME})
        set_target_properties(${DOCA_LIBRARY_NAME}
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${DOCA_INCLUDE_DIR}"
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${DOCA_LIBRARY_${DOCA_LIBRARY_NAME}}"
        )
        message(STATUS "found $DOCA_LIBRARY_NAME as ${DOCA_LIBRARY_${DOCA_LIBRARY_NAME}}")
    endforeach()

    foreach(DOCA_LIBRARY_NAME IN LISTS DOCA_STATIC_LIBS)
        find_library(DOCA_LIBRARY_${DOCA_LIBRARY_NAME} NAMES ${DOCA_LIBRARY_NAME} HINTS ${DOCA_LIB_DIR})
        add_library(${DOCA_LIBRARY_NAME} STATIC IMPORTED GLOBAL)
        add_library(doca::${DOCA_LIBRARY_NAME} ALIAS ${DOCA_LIBRARY_NAME})
        set_target_properties(${DOCA_LIBRARY_NAME}
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${DOCA_INCLUDE_DIR}"
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${DOCA_LIBRARY_${DOCA_LIBRARY_NAME}}"
        )
    endforeach()

    add_library(doca INTERFACE)
    add_library(doca::doca ALIAS doca)
    target_link_libraries(doca
        INTERFACE
            ${DOCA_SHARED_LIBS}
            ${DOCA_STATIC_LIBS}
    )

endif()
