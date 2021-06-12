#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../py_tensor/py_tensor.h"
#include "core/Tensor.h"
#include "core/ops/reduction.h"
#include "numpy/arrayobject.h"

#include "../error_defs.h"
#include "../macros.h"

#define REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims)           \
    {                                                                       \
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist, &t1, \
                                         &axis, &keepdims)) {               \
            PyErr_SetString(PyExc_TypeError,                                \
                            "must pass a tensor and an integer for axis");  \
            return nullptr;                                                 \
        }                                                                   \
    }

/** begin block
 * name = [add, sub, mul, div]
 * op = [+, -, *, /]
 */

RETURN_OBJECT ops_sum(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    PyObject* axis = Py_None;
    int keepdims = 0;
    static char* kwlist[] = {"tensor", "axis", "keepdims", NULL};

    REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims);

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    if (axis == Py_None) {
        ret_class->tensor =
            sail::ops::sum(((PyTensor*)t1)->tensor, NULLDIM, (bool)keepdims);
    } else {
        ret_class->tensor = sail::ops::sum(((PyTensor*)t1)->tensor,
                                           PyLong_AsLong(axis), (bool)keepdims);
    }

    ret_class->ndim = ret_class->tensor.get_ndim();
    ret_class->requires_grad = ret_class->tensor.requires_grad;
    ret_class->dtype = ((PyTensor*)t1)->dtype;
    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}
RETURN_OBJECT ops_max(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    PyObject* axis = Py_None;
    int keepdims = 0;
    static char* kwlist[] = {"tensor", "axis", "keepdims", NULL};

    REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims);

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    if (axis == Py_None) {
        ret_class->tensor =
            sail::ops::max(((PyTensor*)t1)->tensor, NULLDIM, (bool)keepdims);
    } else {
        ret_class->tensor = sail::ops::max(((PyTensor*)t1)->tensor,
                                           PyLong_AsLong(axis), (bool)keepdims);
    }

    ret_class->ndim = ret_class->tensor.get_ndim();
    ret_class->requires_grad = ret_class->tensor.requires_grad;
    ret_class->dtype = ((PyTensor*)t1)->dtype;
    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}

RETURN_OBJECT ops_mean(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    PyObject* axis = Py_None;
    int keepdims = 0;
    static char* kwlist[] = {"tensor", "axis", "keepdims", NULL};

    REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims);

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (axis == Py_None) {
        ret_class->tensor =
            sail::ops::mean(((PyTensor*)t1)->tensor, NULLDIM, (bool)keepdims);
    } else {
        ret_class->tensor = sail::ops::mean(
            ((PyTensor*)t1)->tensor, PyLong_AsLong(axis), (bool)keepdims);
    }

    ret_class->ndim = ret_class->tensor.get_ndim();
    ret_class->requires_grad = ret_class->tensor.requires_grad;
    ret_class->dtype = ((PyTensor*)t1)->dtype;
    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}

/** end block **/
