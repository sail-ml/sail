#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/dtypes.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

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
