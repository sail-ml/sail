#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/dtypes.h"
#include "../../src/modules/modules.h"
#include "../../src/optimizers/optimizers.h"
#include "numpy/arrayobject.h"

#include "../macros.h"
#include "../py_module/py_module.h"

using SCTensor = sail::Tensor;
using Optimizer = sail::optimizers::Optimizer;

typedef struct {
    PyObject_HEAD PyObject *base_object = NULL;
    Optimizer *optimizer = NULL;
} PyOptimizer;

///////////////////// DEFINITIONS ///////////////////////

/////////////// BASE PY FCNS ////////////////
static int PyOptimizer_init(PyOptimizer *self, PyObject *args,
                            PyObject *kwargs);
static int PyOptimizer_traverse(PyOptimizer *self, visitproc visit, void *arg);
static int PyOptimizer_clear(PyOptimizer *self);
static void PyOptimizer_dealloc(PyOptimizer *self);
RETURN_OBJECT PyOptimizer_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwds);

//////////////////////////////////////////////
RETURN_OBJECT PyOptimizer_track_module(PyOptimizer *self, PyObject *args,
                                       PyObject *kwargs);
RETURN_OBJECT PyOptimizer_update(PyOptimizer *self);

static PyMethodDef PyOptimizer_methods[] = {
    {"track_module", (PyCFunction)PyOptimizer_track_module,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"update", (PyCFunction)PyOptimizer_update, METH_NOARGS, NULL},
    {NULL} /* Sentinel */
};

//////////////// TYPE DEF ////////////////////
static PyTypeObject PyOptimizerType = {
    PyVarObject_HEAD_INIT(NULL, 0) "libsail_c.Optimizer", /* tp_name */
    sizeof(PyOptimizer),                                  /* tp_basicsize */
    0,                                                    /* tp_itemsize */
    (destructor)PyOptimizer_dealloc,                      /* tp_dealloc */
    0,                                                    /* tp_print */
    0,                                                    /* tp_getattr */
    0,                                                    /* tp_setattr */
    0,                                                    /* tp_reserved */
    0,                                                    /* tp_repr */
    0,                                                    /* tp_as_number */
    0,                                                    /* tp_as_sequence */
    0,                                                    /* tp_as_mapping */
    0,                                                    /* tp_hash */
    0,                                                    /* tp_call */
    0,                                                    /* tp_str */
    0,                                                    /* tp_getattro */
    0,                                                    /* tp_setattro */
    0,                                                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,             /* tp_flags */
    NULL,                               /* tp_doc */
    (traverseproc)PyOptimizer_traverse, /* tp_traverse */
    (inquiry)PyOptimizer_clear,         /* tp_clear */
    0,                                  /* tp_richcompare */
    0,                                  /* tp_weaklistoffset */
    0,                                  /* tp_iter */
    0,                                  /* tp_iternext */
    PyOptimizer_methods,                /* tp_methods */
    0,                                  /* tp_members */
    0,                          // PyOptimizer_getsetters, /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)PyOptimizer_init, /* tp_init */
    0,                          /* tp_alloc */
    PyOptimizer_new,            /* tp_new */
    PyObject_GC_Del             /* tp_free */

};