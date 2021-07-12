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
#include "core/initializers/initializers.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "numpy/arrayobject.h"

#include "../arg_parser.h"
#include "../error_defs.h"
#include "../macros.h"

static PyObject* sail_kaiming_uniform(PyObject* self, PyObject* args,
                                      PyObject* kwargs) {
    START_EXCEPTION_HANDLING

    // clang-format off
    PythonArgParser<3> parser = PythonArgParser<3>(
        {
            "kaiming_uniform(Tensor x1, string mode = \"fan_in\", string nonlin = \"leaky_relu\")"
        },
    args, kwargs);
    // clang-format on 
    parser.parse();
    sail::initializers::kaiming_uniform(parser.tensor(0), parser.string(1), parser.string(2));

    Py_INCREF((PyObject*)parser.py_tensor(0));
    return (PyObject*)parser.py_tensor(0);

    END_EXCEPTION_HANDLING
}

static PyObject* sail_kaiming_normal(PyObject* self, PyObject* args,
                                      PyObject* kwargs) {
    START_EXCEPTION_HANDLING

    // clang-format off
    PythonArgParser<3> parser = PythonArgParser<3>(
        {
            "kaiming_normal(Tensor x1, string mode = \"fan_in\", string nonlin = \"leaky_relu\")"
        },
    args, kwargs);
    // clang-format on 
    parser.parse();
    sail::initializers::kaiming_normal(parser.tensor(0), parser.string(1), parser.string(2));

    Py_INCREF((PyObject*)parser.py_tensor(0));
    return (PyObject*)parser.py_tensor(0);

    END_EXCEPTION_HANDLING
}

static PyObject* sail_xavier_uniform(PyObject* self, PyObject* args,
                                      PyObject* kwargs) {
    START_EXCEPTION_HANDLING

    // clang-format off
    PythonArgParser<2> parser = PythonArgParser<2>(
        {
            "xavier_uniform(Tensor x1, float gain = 1.0)"
        },
    args, kwargs);
    // clang-format on 
    parser.parse();
    sail::initializers::xavier_uniform(parser.tensor(0), parser.double_(1));

    Py_INCREF((PyObject*)parser.py_tensor(0));
    return (PyObject*)parser.py_tensor(0);

    END_EXCEPTION_HANDLING
}

static PyObject* sail_xavier_normal(PyObject* self, PyObject* args,
                                      PyObject* kwargs) {
    START_EXCEPTION_HANDLING

    // clang-format off
    PythonArgParser<2> parser = PythonArgParser<2>(
        {
            "xavier_normal(Tensor x1, float gain = 1.0)"
        },
    args, kwargs);
    // clang-format on 
    parser.parse();
    sail::initializers::xavier_normal(parser.tensor(0), parser.double_(1));

    Py_INCREF((PyObject*)parser.py_tensor(0));
    return (PyObject*)parser.py_tensor(0);

    END_EXCEPTION_HANDLING
}

static PyMethodDef InitFuncs[] = {
    {"kaiming_uniform", (PyCFunction)sail_kaiming_uniform,
     METH_VARARGS | METH_KEYWORDS},
    {"kaiming_normal", (PyCFunction)sail_kaiming_normal,
     METH_VARARGS | METH_KEYWORDS},
    {"xavier_uniform", (PyCFunction)sail_xavier_uniform,
     METH_VARARGS | METH_KEYWORDS},
    {"xavier_normal", (PyCFunction)sail_xavier_normal,
     METH_VARARGS | METH_KEYWORDS},
    {NULL}};