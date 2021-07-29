#pragma once

#include <Python.h>
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "py_tensor/py_tensor_def.h"

#define RETURN_OBJECT static PyObject *

#define BINARY_TENSOR_TYPE_CHECK(a, b)               \
    {                                                \
        if (!PyObject_TypeCheck(a, &PyTensorType)) { \
            return nullptr;                          \
        }                                            \
        if (!PyObject_TypeCheck(b, &PyTensorType)) { \
            return nullptr;                          \
        }                                            \
    }

#define COPY(src, dest)                      \
    {                                        \
        dest->tensor = src->tensor;          \
        dest->base_object = (PyObject *)src; \
        Py_INCREF(src);                      \
    }
#define SET_BASE(src, dest)                  \
    {                                        \
        Py_INCREF(src);                      \
        dest->base_object = (PyObject *)src; \
    }

#define GENERATE_FROM_TENSOR(pyobj, t) \
    { pyobj->tensor = t; }

#define SEQUENCE_TO_LIST(sequence, list)                                \
    {                                                                   \
        if (sequence == NULL) {                                         \
            list = {1};                                                 \
        } else if (PyLong_Check(sequence)) {                            \
            list = {PyLong_AsLong(sequence)};                           \
        } else {                                                        \
            sequence = PySequence_Tuple(sequence);                      \
            int len = PyTuple_Size(sequence);                           \
            if (len == -1) {                                            \
                list = {PyLong_AsLong(sequence)};                       \
            } else {                                                    \
                while (len--) {                                         \
                    list.push_back(                                     \
                        PyLong_AsLong(PyTuple_GetItem(sequence, len))); \
                }                                                       \
                std::reverse(list.begin(), list.end());                 \
            }                                                           \
        }                                                               \
    }