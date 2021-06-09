#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "../../../src/Tensor.h"
#include "../../../src/ops/ops.h"
#include "../../../src/tensor_shape.h"
#include "../../py_tensor/py_tensor.h"
#include "numpy/arrayobject.h"

#include "../../macros.h"

RETURN_OBJECT ops_pow(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyTensor* t1;
    sail::Tensor t;
    PyTensor* power = NULL;
    static char* kwlist[] = {"base", "power", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &t1, &power)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::ops::power(t1->tensor, power->tensor);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;

    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_exp(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyTensor* t1;
    PyTensor* power = NULL;
    static char* kwlist[] = {"base", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &t1)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::ops::exp(t1->tensor);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;

    return (PyObject*)ret_class;
}
