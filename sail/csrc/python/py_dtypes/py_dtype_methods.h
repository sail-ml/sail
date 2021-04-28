#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/dtypes.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

RETURN_OBJECT
PyDtype_Int32_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyDtype* self;
    self = (PyDtype*)type->tp_alloc(type, 0);
    self->dtype = Dtype::sInt32;
    return (PyObject*)self;
}
RETURN_OBJECT
PyDtype_Float32_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyDtype* self;
    self = (PyDtype*)type->tp_alloc(type, 0);
    self->dtype = Dtype::sFloat32;
    return (PyObject*)self;
}
RETURN_OBJECT
PyDtype_Float64_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyDtype* self;
    self = (PyDtype*)type->tp_alloc(type, 0);
    self->dtype = Dtype::sFloat64;
    return (PyObject*)self;
}
static int PyDtype_traverse(PyDtype* self, visitproc visit, void* arg) {
    Py_VISIT(self->base_object);
    return 0;
}
static int PyDtype_clear(PyDtype* self) { return 0; }
static void PyDtype_dealloc(PyDtype* self) {
    PyObject_GC_UnTrack(self);
    PyDtype_clear(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}
