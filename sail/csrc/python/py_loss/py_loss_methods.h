#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../core/Tensor.h"
#include "../../core/TensorBody.h"
#include "../../core/dtypes.h"
#include "../../core/factories.h"
#include "../../core/ops/ops.h"
#include "../../core/tensor_shape.h"
#include "../../core/types.h"
#include "numpy/arrayobject.h"
#include "py_loss_def.h"

#include "../macros.h"

static int PyLoss_init(PyLoss *self, PyObject *args, PyObject *kwargs) {
    return 0;
}
static int PyLoss_traverse(PyLoss *self, visitproc visit, void *arg) {
    Py_VISIT(self->base_object);
    return 0;
}

static int PyLoss_clear(PyLoss *self) {
    delete self->module;
    if (self->base_object != NULL) {
        Py_DECREF(self->base_object);
    }
    return 0;
}

static void PyLoss_dealloc(PyLoss *self) {
    PyObject_GC_UnTrack(self);
    PyLoss_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

RETURN_OBJECT
PyLoss_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyLoss *self;
    self = (PyLoss *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/////////////////////////////////////////

RETURN_OBJECT PyLoss_forward(PyLoss *self, PyObject *args, PyObject *kwds) {
    PyErr_SetString(PyExc_NotImplementedError, "");
    return NULL;
}
