# This module defines:
#  SentencePiece_FOUND - True if SentencePiece was found
#  SentencePiece_INCLUDE_DIRS - The SentencePiece include directories
#  SentencePiece_LIBRARIES - The libraries needed to use SentencePiece

# Search for the header file
find_path(SentencePiece_INCLUDE_DIR
    NAMES sentencepiece_processor.h # Common header for SentencePiece
    PATHS ${SentencePiece_INCLUDE_DIR}
    NO_DEFAULT_PATH
    DOC "SentencePiece include directory"
)

# --- Intelligent Library Name Search ---
# if (WIN32)
#     set(_sp_names sentencepiece.lib libsentencepiece.a sentencepiece)
# elseif (APPLE)
#     set(_sp_names libsentencepiece.dylib sentencepiece)
# else() # Assume Linux/UNIX
#     set(_sp_names libsentencepiece.so sentencepiece)
# endif()

# Now use the list of names we just generated
#    NAMES ${_sp_names}
find_library(SentencePiece_LIBRARY
    NAMES sentencepiece
    PATHS ${SentencePiece_LIBRARY_DIR}
    NO_DEFAULT_PATH
    DOC "SentencePiece library"
)

# Handle the REQUIRED and QUIET arguments and set SentencePiece_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SentencePiece
    REQUIRED_VARS SentencePiece_LIBRARY SentencePiece_INCLUDE_DIR
    FAIL_MESSAGE "Could NOT find SentencePiece. Set SentencePiece_INCLUDE_DIR and SentencePiece_LIBRARY_DIR."
)

mark_as_advanced(SentencePiece_INCLUDE_DIR SentencePiece_LIBRARY)

# Set the output variables
if(SentencePiece_FOUND)
    set(SentencePiece_LIBRARIES ${SentencePiece_LIBRARY})
    set(SentencePiece_INCLUDE_DIRS ${SentencePiece_INCLUDE_DIR})
endif()
