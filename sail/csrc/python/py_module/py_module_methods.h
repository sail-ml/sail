#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/TensorBody.h"
#include "../../src/dtypes.h"
#include "../../src/factories.h"
#include "../../src/ops/ops.h"
#include "../../src/tensor_shape.h"
#include "../../src/types.h"
#include "numpy/arrayobject.h"
#include "py_module_def.h"

#include "../macros.h"

static int PyModule_init(PyModule *self, PyObject *args, PyObject *kwargs) {
    return 0;
}
static int PyModule_traverse(PyModule *self, visitproc visit, void *arg) {
    Py_VISIT(self->base_object);
    return 0;
}

static int PyModule_clear(PyModule *self) {
    delete self->module;
    if (self->base_object != NULL) {
        Py_DECREF(self->base_object);
    }
    return 0;
}

static void PyModule_dealloc(PyModule *self) {
    PyObject_GC_UnTrack(self);
    PyModule_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

RETURN_OBJECT
PyModule_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyModule *self;
    self = (PyModule *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/////////////////////////////////////////

RETURN_OBJECT PyModule_forward(PyModule *self, PyObject *args, PyObject *kwds) {
    PyErr_SetString(PyExc_NotImplementedError, "");
    return NULL;
}
