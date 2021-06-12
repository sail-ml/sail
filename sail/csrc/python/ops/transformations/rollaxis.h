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
#include "core/ops/reduction.h"
#include "core/tensor_shape.h"
#include "numpy/arrayobject.h"

#include "../../error_defs.h"
#include "../../macros.h"

RETURN_OBJECT ops_rollaxis(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    PyObject* axis = NULL;
    PyObject* position = NULL;
    static char* kwlist[] = {"tensor", "axis", "position", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &t1, &axis,
                                     &position)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    int axis_val = PyLong_AsLong(axis);
    if (position == NULL) {
        ret_class->tensor = sail::ops::rollaxis(t1->tensor, axis_val);
    } else {
        int position_val = PyLong_AsLong(position);
        ret_class->tensor =
            sail::ops::rollaxis(t1->tensor, axis_val, position_val);
    }

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;
    ret_class->base_object = (PyObject*)t1;
    Py_INCREF(ret_class->base_object);

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}

RETURN_OBJECT ops_moveaxis(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    PyObject* axis = NULL;
    PyObject* position = NULL;
    static char* kwlist[] = {"tensor", "axis", "position", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &t1, &axis,
                                     &position)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    int axis_val = PyLong_AsLong(axis);
    if (position == NULL) {
        ret_class->tensor = sail::ops::rollaxis(t1->tensor, axis_val);
    } else {
        int position_val = PyLong_AsLong(position);
        ret_class->tensor =
            sail::ops::moveaxis(t1->tensor, axis_val, position_val);
    }

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;
    ret_class->base_object = (PyObject*)t1;
    Py_INCREF(ret_class->base_object);

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}
