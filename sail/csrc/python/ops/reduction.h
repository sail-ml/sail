#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/ops/reduction.h"
#include "../py_tensor/py_tensor.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

/** begin block
 * name = [add, sub, mul, div]
 * op = [+, -, *, /]
 */

RETURN_OBJECT ops_sum(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyTensor* t1;
    int axis = -1;
    static char* kwlist[] = {"tensor", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &t1, &axis)) {
        PyErr_SetString(PyExc_TypeError,
                        "must pass a tensor and an integer for axis");
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    if (axis == -1) {
        ret_class->tensor = sail::ops::sum(((PyTensor*)t1)->tensor);
    } else {
        ret_class->tensor = sail::ops::sum(((PyTensor*)t1)->tensor, axis);
    }

    ret_class->ndim = ret_class->tensor.get_ndim();
    ret_class->requires_grad = ret_class->tensor.requires_grad;
    ret_class->dtype = ((PyTensor*)t1)->dtype;
    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_mean(PyObject* self, PyObject* args) {
    PyTensor* t1;

    if (!PyArg_ParseTuple(args, "O", &t1)) {
        PyErr_SetString(PyExc_TypeError, "Inputs should be Sail Tensors");
        return NULL;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::ops::mean(((PyTensor*)t1)->tensor);

    ret_class->ndim = ret_class->tensor.get_ndim();
    ret_class->dtype = ((PyTensor*)t1)->dtype;

    return (PyObject*)ret_class;
}

/** end block **/
