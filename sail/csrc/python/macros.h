#pragma once

#include <Python.h>
#include "py_tensor/py_tensor.h"

#define RETURN_OBJECT static PyObject *
// using RETURN_OBJECT = RETURN_OBJECT;

#define BINARY_TENSOR_TYPE_CHECK(a, b)               \
    {                                                \
        if (!PyObject_TypeCheck(a, &PyTensorType)) { \
            return NULL;                             \
        }                                            \
        if (!PyObject_TypeCheck(b, &PyTensorType)) { \
            return NULL;                             \
        }                                            \
    }

#define COPY(src, dest)                   \
    {                                     \
        dest->tensor = src->tensor;       \
        dest->ndim = src->ndim;           \
        dest->dtype = src->dtype;         \
        dest->ob_base = *(PyObject *)src; \
        Py_INCREF(src);                   \
    }
