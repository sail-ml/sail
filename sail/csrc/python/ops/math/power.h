#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "../../py_tensor/py_tensor.h"
#include "core/Tensor.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "numpy/arrayobject.h"

#include "../../error_defs.h"
#include "../../macros.h"

RETURN_OBJECT ops_pow(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    sail::Tensor t;
    PyTensor* power = NULL;
    static char* kwlist[] = {"base", "power", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &t1, &power)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::ops::power(t1->tensor, power->tensor);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}

RETURN_OBJECT ops_exp(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    PyTensor* power = NULL;
    static char* kwlist[] = {"base", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &t1)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::ops::exp(t1->tensor);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}

RETURN_OBJECT ops_log(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    PyTensor* power = NULL;
    static char* kwlist[] = {"x", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &t1)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::ops::log(t1->tensor);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}
