# BASED ON https://github.com/pytorch/pytorch/blob/master/cmake/Modules/FindMKL.cmake 

# - Find INTEL MKL library
#
# This module sets the following variables:
#  MKL_FOUND - set to true if a library implementing the CBLAS interface is found
#  MKL_VERSION - best guess of the found mkl version
#  MKL_INCLUDE_DIR - path to include dir.
#  MKL_LIBRARIES - list of libraries for base mkl
#  MKL_OPENMP_TYPE - OpenMP flavor that the found mkl uses: GNU or Intel
#  MKL_OPENMP_LIBRARY - path to the OpenMP library the found mkl uses
#  MKL_LAPACK_LIBRARIES - list of libraries to add for lapack
#  MKL_SCALAPACK_LIBRARIES - list of libraries to add for scalapack
#  MKL_SOLVER_LIBRARIES - list of libraries to add for the solvers
#  MKL_CDFT_LIBRARIES - list of libraries to add for the solvers

# Do nothing if MKL_FOUND was set before!
IF (NOT MKL_FOUND)

SET(MKL_VERSION)
SET(MKL_INCLUDE_DIR)
SET(MKL_LIBRARIES)
SET(MKL_OPENMP_TYPE)
SET(MKL_OPENMP_LIBRARY)
SET(MKL_LAPACK_LIBRARIES)
SET(MKL_SCALAPACK_LIBRARIES)
SET(MKL_SOLVER_LIBRARIES)
SET(MKL_CDFT_LIBRARIES)

# Includes
INCLUDE(CheckTypeSize)
INCLUDE(CheckFunctionExists)

IF (EXISTS "/opt/intel/oneapi/mkl/latest/")
SET(INTEL_MKL_DIR "/opt/intel/oneapi/mkl/latest")
SET(INTEL_MKL_DIR_NOT_FOUND FALSE)
ELSE()
SET(INTEL_MKL_DIR_NOT_FOUND TRUE)
ENDIF()

if (NOT INTEL_MKL_DIR_NOT_FOUND)
MACRO(FIND_MKLDNN_LIBRARIES LIBRARIES BASE_DIR _list)
SET(_prefix "${BASE_DIR}")
SET(${LIBRARIES})
FOREACH(_library ${_list})
    list(APPEND ${LIBRARIES} "${${_prefix}}/lib/intel64/lib${_library}.so")
ENDFOREACH(_library ${_list})
MARK_AS_ADVANCED(${LIBRARIES})
ENDMACRO(FIND_MKLDNN_LIBRARIES)

FIND_MKLDNN_LIBRARIES(MKL_LIBRARIES INTEL_MKL_DIR "mkl_intel_lp64;mkl_gnu_thread;mkl_core;mkl_rt;")

message(STATUS "MKL LIBS")
message(STATUS ${MKL_LIBRARIES})

set(MKL_INCLUDE_DIR ${INTEL_MKL_DIR}/include)
MARK_AS_ADVANCED(MKL_LIBRARIES)
MARK_AS_ADVANCED(MKL_INCLUDE_DIR)
MARK_AS_ADVANCED(MKL_FOUND)

set(MKL_FOUND TRUE)

endif()

#         FOREACH(mklthread ${mklthreads})
#           IF (NOT MKL_LIBRARIES)
#             CHECK_ALL_LIBRARIES(MKL_LIBRARIES MKL_OPENMP_TYPE MKL_OPENMP_LIBRARY cblas_sgemm
#               "mkl_cdft_core;mkl_${mkliface}${mkl64};${mklthread};mkl_core;${mklrtl};${mkl_pthread};${mkl_m};${mkl_dl}" "")
#           ENDIF (NOT MKL_LIBRARIES)
#         ENDFOREACH(mklthread)
#       ENDFOREACH(mkl64)
#     ENDFOREACH(mkliface)
#   ENDFOREACH(mklrtl)
# ENDIF (NOT "${MKL_THREADING}" STREQUAL "SEQ")

# # Second: search for sequential ones
# FOREACH(mkliface ${mklifaces})
#   FOREACH(mkl64 ${mkl64s} "")
#     IF (NOT MKL_LIBRARIES)
#       CHECK_ALL_LIBRARIES(MKL_LIBRARIES MKL_OPENMP_TYPE MKL_OPENMP_LIBRARY cblas_sgemm
#         "mkl_cdft_core;mkl_${mkliface}${mkl64};mkl_sequential;mkl_core;${mkl_m};${mkl_dl}" "")
#       IF (MKL_LIBRARIES)
#         SET(mklseq "_sequential")
#       ENDIF (MKL_LIBRARIES)
#     ENDIF (NOT MKL_LIBRARIES)
#   ENDFOREACH(mkl64)
# ENDFOREACH(mkliface)

# # First: search for parallelized ones with native pthread lib
# FOREACH(mklrtl ${mklrtls} "")
#   FOREACH(mkliface ${mklifaces})
#     FOREACH(mkl64 ${mkl64s} "")
#       IF (NOT MKL_LIBRARIES)
#         CHECK_ALL_LIBRARIES(MKL_LIBRARIES MKL_OPENMP_TYPE MKL_OPENMP_LIBRARY cblas_sgemm
#           "mkl_cdft_core;mkl_${mkliface}${mkl64};${mklthread};mkl_core;${mklrtl};pthread;${mkl_m};${mkl_dl}" "")
#       ENDIF (NOT MKL_LIBRARIES)
#     ENDFOREACH(mkl64)
#   ENDFOREACH(mkliface)
# ENDFOREACH(mklrtl)

# # Check for older versions
# IF (NOT MKL_LIBRARIES)
#   SET(MKL_VERSION 900)
#   CHECK_ALL_LIBRARIES(MKL_LIBRARIES MKL_OPENMP_TYPE MKL_OPENMP_LIBRARY cblas_sgemm
#     "mkl;guide;pthread;m" "")
# ENDIF (NOT MKL_LIBRARIES)

# # Include files
# IF (MKL_LIBRARIES)
#   FIND_PATH(MKL_INCLUDE_DIR "mkl_cblas.h")
#   MARK_AS_ADVANCED(MKL_INCLUDE_DIR)
# ENDIF (MKL_LIBRARIES)

# # Other libraries
# IF (MKL_LIBRARIES)
#   FOREACH(mkl64 ${mkl64s} "_core" "")
#     FOREACH(mkls ${mklseq} "")
#       IF (NOT MKL_LAPACK_LIBRARIES)
#         FIND_LIBRARY(MKL_LAPACK_LIBRARIES NAMES "mkl_lapack${mkl64}${mkls}")
#         MARK_AS_ADVANCED(MKL_LAPACK_LIBRARIES)
#       ENDIF (NOT MKL_LAPACK_LIBRARIES)
#       IF (NOT MKL_SCALAPACK_LIBRARIES)
#         FIND_LIBRARY(MKL_SCALAPACK_LIBRARIES NAMES "mkl_scalapack${mkl64}${mkls}")
#         MARK_AS_ADVANCED(MKL_SCALAPACK_LIBRARIES)
#       ENDIF (NOT MKL_SCALAPACK_LIBRARIES)
#       IF (NOT MKL_SOLVER_LIBRARIES)
#         FIND_LIBRARY(MKL_SOLVER_LIBRARIES NAMES "mkl_solver${mkl64}${mkls}")
#         MARK_AS_ADVANCED(MKL_SOLVER_LIBRARIES)
#       ENDIF (NOT MKL_SOLVER_LIBRARIES)
#       IF (NOT MKL_CDFT_LIBRARIES)
#         FIND_LIBRARY(MKL_CDFT_LIBRARIES NAMES "mkl_cdft${mkl64}${mkls}")
#         MARK_AS_ADVANCED(MKL_CDFT_LIBRARIES)
#       ENDIF (NOT MKL_CDFT_LIBRARIES)
#     ENDFOREACH(mkls)
#   ENDFOREACH(mkl64)
# ENDIF (MKL_LIBRARIES)

# # Final
# SET(CMAKE_LIBRARY_PATH ${saved_CMAKE_LIBRARY_PATH})
# SET(CMAKE_INCLUDE_PATH ${saved_CMAKE_INCLUDE_PATH})
# IF (MKL_LIBRARIES AND MKL_INCLUDE_DIR)
#   SET(MKL_FOUND TRUE)
# ELSE (MKL_LIBRARIES AND MKL_INCLUDE_DIR)
#   if (MKL_LIBRARIES AND NOT MKL_INCLUDE_DIR)
#     MESSAGE(WARNING "MKL libraries files are found, but MKL header files are \
#       not. You can get them by `conda install mkl-include` if using conda (if \
#       it is missing, run `conda upgrade -n root conda` first), and \
#       `pip install mkl-devel` if using pip. If build fails with header files \
#       available in the system, please make sure that CMake will search the \
#       directory containing them, e.g., by setting CMAKE_INCLUDE_PATH.")
#   endif()
#   SET(MKL_FOUND FALSE)
#   SET(MKL_VERSION)  # clear MKL_VERSION
# ENDIF (MKL_LIBRARIES AND MKL_INCLUDE_DIR)

# # list(APPEND MKL_LIBRARIES )
# FIND_LIBRARY(avx_library NAMES "mkl_avx${mkl64}${mkls}")
# FIND_LIBRARY(avx2_library NAMES "mkl_avx2${mkl64}${mkls}")
# FIND_LIBRARY(def_library NAMES "mkl_def${mkl64}${mkls}")
# FIND_LIBRARY(rt_library NAMES "mkl_rt${mkl64}${mkls}")
# set(MKL_AVX_LIBRARIES "${avx_library};${avx2_library};${def_library};${rt_library}")
# MARK_AS_ADVANCED(MKL_AVX_LIBRARIES)
# # Standard termination
# IF(NOT MKL_FOUND AND MKL_FIND_REQUIRED)
#   MESSAGE(FATAL_ERROR "MKL library not found. Please specify library location \
#     by appending the root directory of the MKL installation to the environment variable CMAKE_PREFIX_PATH.")
# ENDIF(NOT MKL_FOUND AND MKL_FIND_REQUIRED)
# IF(NOT MKL_FIND_QUIETLY)
#   IF(MKL_FOUND)
#     MESSAGE(STATUS "MKL library found")
#   ELSE(MKL_FOUND)
#     MESSAGE(STATUS "MKL library not found")
#   ENDIF(MKL_FOUND)
# ENDIF(NOT MKL_FIND_QUIETLY)

# # Do nothing if MKL_FOUND was set before!
ENDIF (NOT MKL_FOUND)
