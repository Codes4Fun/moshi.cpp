# This module defines:
#  GGML_FOUND - True if GGML was found
#  GGML_INCLUDE_DIRS - The GGML include directories
#  GGML_LIBRARIES - The libraries needed to use GGML

message(STATUS "DEBUG: User-provided GGML_INCLUDE_DIR is: ${GGML_INCLUDE_DIR}")
message(STATUS "DEBUG: User-provided GGML_LIBRARY_DIR is: ${GGML_LIBRARY_DIR}")

# Search for the header file 'ggml.h' in the paths provided by the user.
# The user will set GGML_INCLUDE_DIR with -DGGML_INCLUDE_DIR=/path/to/headers
find_path(GGML_INCLUDE_DIR
    NAMES ggml.h
    PATHS ${GGML_INCLUDE_DIR}
    NO_DEFAULT_PATH # Only search the user-provided path
    DOC "GGML include directory"
)

message(STATUS "DEBUG: find_path for GGML_INCLUDE_DIR found: ${GGML_INCLUDE_DIR}")

# Search for the library file 'libggml.so' (or .a, .dylib, .lib) in the paths provided by the user.
# The user will set GGML_LIBRARY_DIR with -DGGML_LIBRARY_DIR=/path/to/libs
find_library(GGML_LIBRARY
    NAMES ggml
    PATHS ${GGML_LIBRARY_DIR}
    NO_DEFAULT_PATH # Only search the user-provided path
    DOC "GGML library"
)

find_library(GGML_BASE_LIBRARY
    NAMES ggml-base
    PATHS ${GGML_LIBRARY_DIR}
    NO_DEFAULT_PATH # Only search the user-provided path
    DOC "GGML library"
)

message(STATUS "DEBUG: find_library for GGML_LIBRARY found: ${GGML_LIBRARY}")

# Handle the REQUIRED and QUIET arguments to find_package()
# and set GGML_FOUND to TRUE if all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GGML
    REQUIRED_VARS GGML_LIBRARY GGML_BASE_LIBRARY GGML_INCLUDE_DIR
    FAIL_MESSAGE "Could NOT find GGML. Set GGML_INCLUDE_DIR and GGML_LIBRARY_DIR."
)

# Mark the following variables as advanced so they don't show up in ccmake/cmake-gui by default.
mark_as_advanced(GGML_INCLUDE_DIR GGML_LIBRARY)

# Set the output variables for the user of this module.
if(GGML_FOUND)
    set(GGML_LIBRARIES ${GGML_LIBRARY} ${GGML_BASE_LIBRARY})
    set(GGML_INCLUDE_DIRS ${GGML_INCLUDE_DIR})
endif()
