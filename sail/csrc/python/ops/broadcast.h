#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "../py_tensor/py_tensor.h"
#include "core/Tensor.h"
#include "core/ops/ops.h"
#include "core/ops/reduction.h"
#include "core/tensor_shape.h"
#include "numpy/arrayobject.h"

#include "../error_defs.h"
#include "../macros.h"

RETURN_OBJECT ops_broadcast_to(PyObject* self, PyObject* args) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    PyObject* tuple;

    if (!PyArg_ParseTuple(args, "OO", &t1, &tuple)) {
        PyErr_SetString(PyExc_TypeError, "Inputs should be Sail Tensors");
        return nullptr;
    }

    int len = PyTuple_Size(tuple);
    if (len == -1) {
        PyErr_SetString(PyExc_TypeError, "Shape must have atleat 1 element.");
        return nullptr;
    }
    std::vector<long> shape;
    while (len--) {
        shape.push_back(PyLong_AsLong(PyTuple_GetItem(tuple, len)));
    }
    std::reverse(shape.begin(), shape.end());

    sail::TensorShape s = sail::TensorShape(shape);

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::ops::broadcast_to(t1->tensor, s);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;
    SET_BASE(t1, ret_class);
    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}
