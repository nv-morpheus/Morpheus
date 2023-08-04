
include(FindPackageHandleStandardArgs)

set(components "ucx")

find_package(PkgConfig QUIET)
pkg_check_modules(PC_UCX ${components})

# message(STATUS "PC_UCX_FOUND: ${PC_UCX_FOUND}")
# message(STATUS "PC_UCX_LIBRARIES: ${PC_UCX_LIBRARIES}")
# message(STATUS "PC_UCX_LINK_LIBRARIES: ${PC_UCX_LINK_LIBRARIES}")
# message(STATUS "PC_UCX_LIBRARY_DIRS: ${PC_UCX_LIBRARY_DIRS}")
# message(STATUS "PC_UCX_LDFLAGS: ${PC_UCX_LDFLAGS}")
# message(STATUS "PC_UCX_LDFLAGS_OTHER: ${PC_UCX_LDFLAGS_OTHER}")
# message(STATUS "PC_UCX_INCLUDE_DIRS: ${PC_UCX_INCLUDE_DIRS}")
# message(STATUS "PC_UCX_CFLAGS: ${PC_UCX_CFLAGS}")
# message(STATUS "PC_UCX_CFLAGS_OTHER: ${PC_UCX_CFLAGS_OTHER}")

# set(mod_prefix "PC_UCX")

# message(STATUS "${mod_prefix}_VERSION: ${${mod_prefix}_VERSION}")
# message(STATUS "${mod_prefix}_PREFIX: ${${mod_prefix}_PREFIX}")
# message(STATUS "${mod_prefix}_INCLUDEDIR: ${${mod_prefix}_INCLUDEDIR}")
# message(STATUS "${mod_prefix}_LIBDIR: ${${mod_prefix}_LIBDIR}")

set(ucx_VERSION ${PC_UCX_VERSION})

find_package_handle_standard_args(ucx
  FOUND_VAR ucx_FOUND
  REQUIRED_VARS
    PC_UCX_FOUND
  VERSION_VAR ucx_VERSION
)

if (UCX_FOUND)
  set(all_ucx_targets "")

  foreach(ucx_library IN ZIP_LISTS PC_UCX_LIBRARIES PC_UCX_LINK_LIBRARIES)
    if (NOT TARGET ucx::${ucx_library_0})
      add_library(ucx::${ucx_library_0} UNKNOWN IMPORTED)
      set_target_properties(ucx::${ucx_library_0} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PC_UCX_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "${PC_UCX_CFLAGS_OTHER}"
        INTERFACE_LINK_OPTIONS "${PC_UCX_LDFLAGS_OTHER}"
        IMPORTED_LOCATION "${ucx_library_1}"
      )
    endif()

    # Add to the list of child targets
    list(APPEND all_ucx_targets "ucx::${ucx_library_0}")
  endforeach()

  if (NOT TARGET ucx::ucx)
    # Combined ucx::ucx target
    add_library(ucx::ucx INTERFACE IMPORTED GLOBAL)
    set_target_properties(ucx::ucx PROPERTIES
      INTERFACE_LINK_LIBRARIES "${all_ucx_targets}"
    )
    endif()
endif()
