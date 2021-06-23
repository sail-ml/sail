#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "../py_tensor/py_tensor.h"
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/ops/reduction.h"
#include "core/tensor_shape.h"
#include "numpy/arrayobject.h"

#include "../error_defs.h"
#include "../macros.h"

RETURN_OBJECT ops_random_uniform(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyObject* shape = NULL;
    double min = 0;
    double max = 1;
    static char* kwlist[] = {"min", "max", "shape", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ddO", kwlist, &min, &max,
                                     &shape)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    Dtype dt = default_dtype;

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    std::vector<long> t_shape;
    if (shape == NULL) {
        t_shape = {1};
    } else if (PyLong_Check(shape)) {
        t_shape = {PyLong_AsLong(shape)};
    } else {
        shape = PySequence_Tuple(shape);
        int len = PyTuple_Size(shape);
        if (len == -1) {
            t_shape = {PyLong_AsLong(shape)};
        } else {
            while (len--) {
                t_shape.push_back(PyLong_AsLong(PyTuple_GetItem(shape, len)));
            }
            std::reverse(t_shape.begin(), t_shape.end());
        }
    }

    ret_class->tensor =
        sail::random::uniform(sail::TensorShape(t_shape), dt, min, max);

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}

RETURN_OBJECT ops_random_uniform_like(PyObject* self, PyObject* args,
                                      PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1 = NULL;
    double min = 0;
    double max = 1;
    static char* kwlist[] = {"tensor", "min", "max", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|dd", kwlist, &t1, &min,
                                     &max)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::random::uniform_like(t1->tensor, min, max);

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}

RETURN_OBJECT ops_random_normal(PyObject* self, PyObject* args,
                                PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyObject* shape = NULL;
    double mean = 0;
    double std = 1;
    static char* kwlist[] = {"mean", "std", "shape", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ddO", kwlist, &mean, &std,
                                     &shape)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    Dtype dt = default_dtype;
    std::vector<long> t_shape;
    if (shape == NULL) {
        t_shape = {1};
    } else if (PyLong_Check(shape)) {
        t_shape = {PyLong_AsLong(shape)};
    } else {
        shape = PySequence_Tuple(shape);
        int len = PyTuple_Size(shape);
        if (len == -1) {
            t_shape = {PyLong_AsLong(shape)};
        } else {
            while (len--) {
                t_shape.push_back(PyLong_AsLong(PyTuple_GetItem(shape, len)));
            }
            std::reverse(t_shape.begin(), t_shape.end());
        }
    }

    ret_class->tensor =
        sail::random::normal(sail::TensorShape(t_shape), mean, std);

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}

RETURN_OBJECT ops_random_normal_like(PyObject* self, PyObject* args,
                                     PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor* t1 = NULL;
    double mean = 0;
    double std = 1;
    static char* kwlist[] = {"tensor", "mean", "std", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|dd", kwlist, &t1, &mean,
                                     &std)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::random::normal_like(t1->tensor, mean, std);

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}
