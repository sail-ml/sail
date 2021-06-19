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

RETURN_OBJECT ops_clip(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1;
    double min, max;
    static char* kwlist[] = {"tensor", "min", "max", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Odd", kwlist, &t1, &min,
                                     &max)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::ops::clip(t1->tensor, min, max);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}
