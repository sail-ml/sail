#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "../../../src/Tensor.h"
#include "../../../src/dtypes.h"
#include "../../../src/factories.h"
#include "../../../src/ops/ops.h"
#include "../../../src/ops/reduction.h"
#include "../../../src/tensor_shape.h"
#include "../../py_tensor/py_tensor.h"
#include "numpy/arrayobject.h"

#include "../../macros.h"

RETURN_OBJECT ops_random_uniform(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
    PyObject* shape = NULL;
    int min = 0;
    int max = 1;
    static char* kwlist[] = {"min", "max", "shape", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiO", kwlist, &min, &max,
                                     &shape)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }

    Dtype dt = Dtype::sFloat64;

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    shape = PySequence_Tuple(shape);
    std::vector<long> t_shape;
    int len = PyTuple_Size(shape);
    if (len == -1) {
        t_shape = {PyLong_AsLong(shape)};
    } else {
        while (len--) {
            t_shape.push_back(PyLong_AsLong(PyTuple_GetItem(shape, len)));
        }
        std::reverse(t_shape.begin(), t_shape.end());
    }

    ret_class->tensor =
        sail::random::uniform(sail::TensorShape(t_shape), dt, min, max);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = get_np_type_numFromDtype(dt);
    ret_class->requires_grad = false;

    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_random_uniform_like(PyObject* self, PyObject* args,
                                      PyObject* kwargs) {
    PyTensor* t1 = NULL;
    int min = 0;
    int max = 1;
    static char* kwlist[] = {"tensor", "min", "max", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", kwlist, &t1, &min,
                                     &max)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::random::uniform_like(t1->tensor, min, max);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->dtype;
    ret_class->requires_grad = t1->requires_grad;

    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_random_normal(PyObject* self, PyObject* args,
                                PyObject* kwargs) {
    PyObject* shape = NULL;
    int mean = 0;
    int std = 1;
    static char* kwlist[] = {"mean", "std", "shape", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiO", kwlist, &mean, &std,
                                     &shape)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }

    Dtype dt = Dtype::sFloat64;

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    shape = PySequence_Tuple(shape);
    std::vector<long> t_shape;
    int len = PyTuple_Size(shape);
    if (len == -1) {
        t_shape = {PyLong_AsLong(shape)};
    } else {
        while (len--) {
            t_shape.push_back(PyLong_AsLong(PyTuple_GetItem(shape, len)));
        }
        std::reverse(t_shape.begin(), t_shape.end());
    }

    ret_class->tensor =
        sail::random::normal(sail::TensorShape(t_shape), dt, mean, std);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = get_np_type_numFromDtype(dt);
    ret_class->requires_grad = false;

    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_random_normal_like(PyObject* self, PyObject* args,
                                     PyObject* kwargs) {
    PyTensor* t1 = NULL;
    int mean = 0;
    int std = 1;
    static char* kwlist[] = {"tensor", "mean", "std", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", kwlist, &t1, &mean,
                                     &std)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::random::normal_like(t1->tensor, mean, std);

    ret_class->ndim = ret_class->tensor.get_shape().ndim();
    ret_class->dtype = t1->dtype;
    ret_class->requires_grad = t1->requires_grad;

    return (PyObject*)ret_class;
}
