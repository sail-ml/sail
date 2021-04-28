#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/dtypes.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

typedef struct {
    PyObject_HEAD PyObject *base_object = NULL;
    Dtype dtype;
} PyDtype;

/////////////// BASE PY FCNS ////////////////
RETURN_OBJECT PyDtype_Int32_new(PyTypeObject *type, PyObject *args,
                                PyObject *kwds);
RETURN_OBJECT PyDtype_Float32_new(PyTypeObject *type, PyObject *args,
                                  PyObject *kwds);
RETURN_OBJECT PyDtype_Float64_new(PyTypeObject *type, PyObject *args,
                                  PyObject *kwds);

static int PyDtype_traverse(PyDtype *self, visitproc visit, void *arg);
static int PyDtype_clear(PyDtype *self);
static void PyDtype_dealloc(PyDtype *self);
