#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "../../../core/Tensor.h"
#include "../../../core/ops/ops.h"
#include "../../../core/ops/reduction.h"
#include "../../../core/tensor_shape.h"
#include "../../py_tensor/py_tensor.h"
#include "numpy/arrayobject.h"

#include "../../macros.h"

RETURN_OBJECT ops_transpose(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyTensor* t1;
    PyObject* tuple = NULL;
    static char* kwlist[] = {"tensor", "axes", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &t1,
                                     &tuple)) {
        PyErr_SetString(PyExc_TypeError, "must pass a tensor and a tuple");
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    if (tuple == NULL) {
        ret_class->tensor = sail::ops::transpose(t1->tensor);
    } else {
        tuple = PySequence_Tuple(tuple);
        int len = PyTuple_Size(tuple);
        if (len == -1) {
            PyErr_SetString(PyExc_TypeError,
                            "Shape must have atleat 1 element.");
        }
        std::vector<long> shape;
        while (len--) {
            shape.push_back(PyLong_AsLong(PyTuple_GetItem(tuple, len)));
        }
        std::reverse(shape.begin(), shape.end());

        ret_class->tensor = sail::ops::transpose(t1->tensor, shape);
    }

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;
    ret_class->base_object = (PyObject*)t1;
    Py_INCREF(ret_class->base_object);

    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_reshape(PyObject* self, PyObject* args) {
    PyTensor* t1;
    PyObject* py_tuple;

    if (!PyArg_ParseTuple(args, "OO", &t1, &py_tuple)) {
        PyErr_SetString(PyExc_TypeError, "must pass a tensor and a shape");
    }

    // BINARY_TENSOR_TYPE_CHECK(t1, t2);
    py_tuple = PySequence_Tuple(py_tuple);
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

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = t1->tensor.expand_dims(dim);
    ret_class->ndim = ret_class->tensor.get_ndim();
    ret_class->dtype = t1->dtype;
    ret_class->base_object = (PyObject*)t1;
    Py_INCREF(t1);

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

    ret_class->tensor = t1->tensor.squeeze(dim);
    ret_class->ndim = ret_class->tensor.get_ndim();
    ret_class->dtype = t1->dtype;
    ret_class->base_object = (PyObject*)t1;
    Py_INCREF(t1);

    return (PyObject*)ret_class;
}