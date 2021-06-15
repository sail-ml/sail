#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/dtypes.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

typedef struct {
    PyObject_HEAD PyObject *base_object = NULL;
    Dtype dtype;
    int dt_val;
} PyDtype;

///////////// REPR ////////////////////////////
RETURN_OBJECT
PyDtype_Repr(PyDtype *obj) {
    if (obj->dtype == Dtype::sInt32) {
        return PyUnicode_FromString("<class 'sail.int32'>");
    } else if (obj->dtype == Dtype::sFloat32) {
        return PyUnicode_FromString("<class 'sail.float32'>");
    } else if (obj->dtype == Dtype::sFloat32) {
        return PyUnicode_FromString("<class 'sail.float64'>");
    } else {
        return PyUnicode_FromString("<class 'sail.dtype'>");
    }
}

static int PyDtype_traverse(PyDtype *self, visitproc visit, void *arg);
static int PyDtype_clear(PyDtype *self);
static void PyDtype_dealloc(PyDtype *self);

static PyTypeObject PyDtypeBase = {
    PyVarObject_HEAD_INIT(NULL, 0) "sail.dtype", /* tp_name */
    sizeof(PyDtype),                                   /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)PyDtype_dealloc,                       /* tp_dealloc */
    0,                                                 /* tp_print */
    0,                                                 /* tp_getattr */
    0,                                                 /* tp_setattr */
    0,                                                 /* tp_reserved */
    &PyDtype_Repr,                                                 /* tp_repr */
    0,                                                 /* tp_as_number */
    0,                                                 /* tp_as_sequence */
    0,                                                 /* tp_as_mapping */
    0,                                                 /* tp_hash */
    0,                                                 /* tp_call */
    0,                                                 /* tp_str */
    0,                                                 /* tp_getattro */
    0,                                                 /* tp_setattro */
    0,                                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,         /* tp_flags */
    "Custom objects",               /* tp_doc */
    (traverseproc)PyDtype_traverse, /* tp_traverse */
    (inquiry)PyDtype_clear,         /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,  // (getiterfunc)Chromosone_getiter, /* tp_iter */
    0,  /* tp_iter */
    0,  /* tp_iternext */
    0,  /* tp_methods */
    0,  /* tp_members */
    0,  // PyTensor_getsetters, /* tp_getset */
    0,  /* tp_base */
    0,  /* tp_dict */
    0,  /* tp_descr_get */
    0,  /* tp_descr_set */
    0,  /* tp_dictoffset */
    0,  //(initproc)PyDtype_init, /* tp_init */
    0,  /* tp_alloc */
    0,
};
