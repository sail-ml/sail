#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "core/modules/modules.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

using SCTensor = sail::Tensor;
using Module = sail::modules::Module;

typedef struct {
    PyObject_HEAD PyObject *base_object = NULL;
    Module *module = NULL;
} PyModule;

///////////////////// DEFINITIONS ///////////////////////

/////////////// BASE PY FCNS ////////////////
static int PyModule_init(PyModule *self, PyObject *args, PyObject *kwargs);
static int PyModule_traverse(PyModule *self, visitproc visit, void *arg);
static int PyModule_clear(PyModule *self);
static void PyModule_dealloc(PyModule *self);
static PyObject *PyModule_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwds);

//////////////////////////////////////////////
static PyObject *PyModule_forward(PyModule *self, PyObject *args,
                                  PyObject *kwds);
static PyObject *PyModule_call(PyModule *self, PyObject *args, PyObject *kwds);
int PyModule_setattr(PyModule *self, PyObject *attr, PyObject *value);

static PyMethodDef PyModule_methods[] = {
    {"forward", (PyCFunction)PyModule_forward, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {NULL} /* Sentinel */
};

//////////////// TYPE DEF ////////////////////
static PyTypeObject PyModuleType = {
    PyVarObject_HEAD_INIT(NULL, 0) "libsail_c.Module", /* tp_name */
    sizeof(PyModule),                                  /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)PyModule_dealloc,                      /* tp_dealloc */
    0,                                                 /* tp_print */
    0,                                                 /* tp_getattr */
    0,                                                 /* tp_setattr */
    0,                                                 /* tp_reserved */
    0,                                                 /* tp_repr */
    0,                                                 /* tp_as_number */
    0,                                                 /* tp_as_sequence */
    0,                                                 /* tp_as_mapping */
    0,                                                 /* tp_hash */
    PyModule_call,                                     /* tp_call */
    0,                                                 /* tp_str */
    PyObject_GenericGetAttr,                           /* tp_getattro */
    PyModule_setattr,                                  /* tp_setattro */
    0,                                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,          /* tp_flags */
    NULL,                            /* tp_doc */
    (traverseproc)PyModule_traverse, /* tp_traverse */
    (inquiry)PyModule_clear,         /* tp_clear */
    0,                               /* tp_richcompare */
    0,                               /* tp_weaklistoffset */
    0,                               /* tp_iter */
    0,                               /* tp_iternext */
    PyModule_methods,                /* tp_methods */
    0,                               /* tp_members */
    0,                               // PyModule_getsetters, /* tp_getset */
    0,                               /* tp_base */
    0,                               /* tp_dict */
    0,                               /* tp_descr_get */
    0,                               /* tp_descr_set */
    0,                               /* tp_dictoffset */
    (initproc)PyModule_init,         /* tp_init */
    0,                               /* tp_alloc */
    PyModule_new,                    /* tp_new */
    PyObject_GC_Del                  /* tp_free */

};