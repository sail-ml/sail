#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "../../../src/Tensor.h"
#include "../../../src/ops/ops.h"
#include "../../../src/ops/reduction.h"
#include "../../../src/tensor_shape.h"
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

        std::cout << getVectorString(shape) << std::endl;

        ret_class->tensor = sail::ops::transpose(t1->tensor, shape);
    }

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->ndim;
    ret_class->requires_grad = t1->requires_grad;
    ret_class->base_object = (PyObject*)t1;
    Py_INCREF(ret_class->base_object);

    return (PyObject*)ret_class;
}
