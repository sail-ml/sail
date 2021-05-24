# set(CMAKE_CXX_COMPILER "gcc")
cmake_minimum_required(VERSION 2.8.12)
project(sail_c LANGUAGES CXX C) # CUDA
# enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)

ADD_DEFINITIONS(-DUSE_AVX2) # NEED FOR AVX2 SUPPORT IN KERNELS

# find_package(Doxygen)

# if (DOXYGEN_FOUND)
#     # set input and output files
#     set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/cpp/source/Doxyfile.in)
#     set(DOXYGEN_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/cpp/Doxyfile-c)
#     # set(DOXYGEN_C_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/cpp/Doxyfile-c)


#     # request to configure the file
#     configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
#     message("Doxygen build started")
#     # message(${DOXYGEN_EXECUTABLE})
#     # message(${DOXYGEN_OUT})
#     # message(${CMAKE_CURRENT_BINARY_DIR})

#     # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
#     add_custom_target( docs ALL
#         COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
#         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
#         COMMENT "Generating API documentation with Doxygen"
#         VERBATIM )
# else (DOXYGEN_FOUND)
#   message("Doxygen need to be installed to generate the doxygen documentation")
# endif (DOXYGEN_FOUND)

add_subdirectory(sail/csrc)