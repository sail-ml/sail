# - Try to find MKLDNN
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#
# The following are set after configuration is done:
#  MKLDNN_FOUND          : set to true if mkl-dnn is found.
#  MKLDNN_INCLUDE_DIR    : path to mkl-dnn include dir.
#  MKLDNN_LIBRARIES      : list of libraries for mkl-dnn
#
# The following variables are used:
#  MKLDNN_USE_NATIVE_ARCH : Whether native CPU instructions should be used in MKLDNN. This should be turned off for
#  general packaging to avoid incompatible CPU instructions. Default: OFF.

IF (NOT MKLDNN_FOUND)

SET(MKLDNN_LIBRARIES)
SET(MKLDNN_INCLUDE_DIR)

IF (EXISTS "/opt/intel/oneapi/dnnl/latest/")
SET(INTEL_DNNL_DIR "/opt/intel/oneapi/dnnl/latest/cpu_iomp")
SET(INTEL_DNNL_DIR_NOT_FOUND FALSE)
ELSE()
SET(INTEL_DNNL_DIR_NOT_FOUND TRUE)
ENDIF()

MACRO(FIND_MKLDNN_LIBRARIES LIBRARIES BASE_DIR _list)
SET(_prefix "${BASE_DIR}")
SET(${LIBRARIES})
FOREACH(_library ${_list})
    list(APPEND ${LIBRARIES} "${${_prefix}}/lib/lib${_library}.so")
ENDFOREACH(_library ${_list})
MARK_AS_ADVANCED(${LIBRARIES})
ENDMACRO(FIND_MKLDNN_LIBRARIES)

FIND_MKLDNN_LIBRARIES(MKLDNN_LIBRARIES INTEL_DNNL_DIR "mkldnn;dnnl")

message(STATUS "LIBS")
message(STATUS ${MKLDNN_LIBRARIES})

set(MKLDNN_INCLUDE_DIR ${INTEL_DNNL_DIR}/include)
MARK_AS_ADVANCED(MKLDNN_LIBRARIES)
MARK_AS_ADVANCED(MKLDNN_INCLUDE_DIR)
MARK_AS_ADVANCED(MKLDNN_FOUND)

set(MKLDNN_FOUND TRUE)

ENDIF(NOT MKLDNN_FOUND)