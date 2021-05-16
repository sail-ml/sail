#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include "../../../src/Tensor.h"
#include "../../../src/ops/ops.h"
#include "../../../src/tensor_shape.h"
#include "../../../src/types.h"
#include "../../py_tensor/py_tensor.h"
#include "numpy/arrayobject.h"

#include "../../macros.h"

RETURN_OBJECT ops_reshape(PyObject* self, PyObject* args) {
    PyTensor* t1;
    PyObject* py_tuple;

    if (!PyArg_ParseTuple(args, "OO", &t1, &py_tuple)) {
        PyErr_SetString(PyExc_TypeError, "must pass a tensor and a shape");
    }

    // BINARY_TENSOR_TYPE_CHECK(t1, t2);
    int len = PyTuple_Size(py_tuple);
    if (len == -1) {
        PyErr_SetString(PyExc_TypeError, "Shape must have atleat 1 element.");
    }
    TensorSize size;
    while (len--) {
        size.push_back(PyLong_AsLong(PyTuple_GetItem(py_tuple, len)));
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    std::reverse(size.begin(), size.end());

    sail::TensorShape new_ = sail::TensorShape(size);

    ret_class->tensor = t1->tensor.reshape(new_);
    ret_class->ndim = t1->tensor.get_ndim();
    ret_class->dtype = t1->dtype;
    ret_class->base_object = (PyObject*)t1;
    Py_INCREF(t1);

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

    if (dim < -1 || dim > t1->tensor.get_ndim()) {
        PyErr_SetString(PyExc_ValueError,
                        ("dim must be in the range of [-1, ndim]"));
        return NULL;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    COPY(t1, ret_class);

    ret_class->tensor.expand_dims(dim);
    ret_class->ndim = ret_class->ndim + 1;

    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_squeeze(PyObject* self, PyObject* args) {
    PyTensor* t1;
    int dim;

    if (!PyArg_ParseTuple(args, "Oi", &t1, &dim)) {
        PyErr_SetString(PyExc_TypeError,
                        "Inputs should be a sail tensor and an integer");
        return NULL;
    }

    if (dim < -1 || dim > t1->tensor.get_ndim()) {
        PyErr_SetString(PyExc_ValueError,
                        ("dim must be in the range of [-1, ndim]"));
        return NULL;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    COPY(t1, ret_class);

    ret_class->tensor.squeeze(dim);
    ret_class->ndim = ret_class->ndim + 1;

    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_matmul(PyObject* self, PyObject* args) {
    PyObject* t1;
    PyObject* t2;

    sail::Tensor tensor1;
    sail::Tensor tensor2;

    if (!PyArg_ParseTuple(args, "OO", &t1, &t2)) {
        PyErr_SetString(PyExc_TypeError, "Inputs should be Sail Tensors");
        return NULL;
    }

    tensor1 = ((PyTensor*)t1)->tensor;
    tensor2 = ((PyTensor*)t2)->tensor;

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    sail::Tensor res = sail::ops::matmul(tensor1, tensor2);

    ret_class->tensor = res;
    ret_class->ndim = ((PyTensor*)t1)->ndim;
    ret_class->dtype = ((PyTensor*)t1)->dtype;

    return (PyObject*)ret_class;
}
