#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include <algorithm>
#include "../../../src/Tensor.h"
#include "../../../src/ops/elementwise.h"
#include "../../../src/types.h"
#include "../../py_tensor/py_tensor.h"
#include "numpy/arrayobject.h"

#include "../../macros.h"

RETURN_OBJECT ops_reshape(PyObject* self, PyObject* args) {
    PyTensor* t1;
    PyObject* py_tuple;

    if (!PyArg_ParseTuple(args, "OO", &t1, &py_tuple)) {
        PyErr_SetString(PyExc_TypeError, "Inputs should be Sail Tensors");
        return NULL;
    }

    // BINARY_TENSOR_TYPE_CHECK(t1, t2);
    int len = PyTuple_Size(py_tuple);
    TensorSize size;
    while (len--) {
        size.push_back(PyLong_AsLong(PyTuple_GetItem(py_tuple, len)));
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    std::reverse(size.begin(), size.end());

    COPY(t1, ret_class);

    ret_class->tensor.reshape(size);

    ret_class->ndim = ret_class->tensor.storage.ndim;

    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_expand_dims(PyObject* self, PyObject* args) {
    PyTensor* t1;
    int dim;

    if (!PyArg_ParseTuple(args, "Oi", &t1, &dim)) {
        PyErr_SetString(PyExc_TypeError,
                        "Inputs should be a sail tensor and an integer");
        return NULL;
    }

    if (dim < -1 || dim > t1->tensor.storage.ndim) {
        PyErr_SetString(
            PyExc_ValueError,
            ("dim must be in the range of [-1, %s]", t1->tensor.storage.ndim));
        return NULL;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    COPY(t1, ret_class);

    ret_class->tensor.expand_dims(dim);
    ret_class->ndim = ret_class->ndim + 1;

    return (PyObject*)ret_class;
}
