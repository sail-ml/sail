#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/dtypes.h"
#include "../../src/modules/modules.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

using SCTensor = sail::Tensor;
using Module = sail::modules::Module;

typedef struct {
    PyObject_HEAD PyObject *base_object = NULL;
    Module *module = NULL;
} PyLoss;

///////////////////// DEFINITIONS ///////////////////////

/////////////// BASE PY FCNS ////////////////
static int PyLoss_init(PyLoss *self, PyObject *args, PyObject *kwargs);
static int PyLoss_traverse(PyLoss *self, visitproc visit, void *arg);
static int PyLoss_clear(PyLoss *self);
static void PyLoss_dealloc(PyLoss *self);
RETURN_OBJECT PyLoss_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

//////////////////////////////////////////////
RETURN_OBJECT PyLoss_forward(PyLoss *self, PyObject *args, PyObject *kwds);

static PyMethodDef PyLoss_methods[] = {
    {"forward", (PyCFunction)PyLoss_forward, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {NULL} /* Sentinel */
};

//////////////// TYPE DEF ////////////////////
static PyTypeObject PyLossType = {
    PyVarObject_HEAD_INIT(NULL, 0) "libsail_c.Module", /* tp_name */
    sizeof(PyLoss),                                    /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)PyLoss_dealloc,                        /* tp_dealloc */
    0,                                                 /* tp_print */
    0,                                                 /* tp_getattr */
    0,                                                 /* tp_setattr */
    0,                                                 /* tp_reserved */
    0,                                                 /* tp_repr */
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
        Py_TPFLAGS_HAVE_GC,        /* tp_flags */
    NULL,                          /* tp_doc */
    (traverseproc)PyLoss_traverse, /* tp_traverse */
    (inquiry)PyLoss_clear,         /* tp_clear */
    0,                             /* tp_richcompare */
    0,                             /* tp_weaklistoffset */
    0,                             /* tp_iter */
    0,                             /* tp_iternext */
    PyLoss_methods,                /* tp_methods */
    0,                             /* tp_members */
    0,                             // PyLoss_getsetters, /* tp_getset */
    0,                             /* tp_base */
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    0,                             /* tp_dictoffset */
    (initproc)PyLoss_init,         /* tp_init */
    0,                             /* tp_alloc */
    PyLoss_new,                    /* tp_new */
    PyObject_GC_Del                /* tp_free */

};