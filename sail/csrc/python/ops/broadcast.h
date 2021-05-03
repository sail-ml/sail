#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/ops/ops.h"
#include "../../src/ops/reduction.h"
#include "../../src/tensor_shape.h"
#include "../py_tensor/py_tensor.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

RETURN_OBJECT ops_broadcast_to(PyObject* self, PyObject* args) {
    PyTensor* t1;
    PyObject* tuple;

    if (!PyArg_ParseTuple(args, "OO", &t1, &tuple)) {
        PyErr_SetString(PyExc_TypeError, "Inputs should be Sail Tensors");
        return NULL;
    }

    int len = PyTuple_Size(tuple);
    std::vector<long> shape;
    while (len--) {
        shape.push_back(PyLong_AsLong(PyTuple_GetItem(tuple, len)));
    }

    sail::TensorShape s = sail::TensorShape(shape);

    t1->tensor = sail::ops::broadcast_to(t1->tensor, s);

    return (PyObject*)t1;
}
