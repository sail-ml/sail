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
#include "../py_module/py_module.h"
#include "numpy/arrayobject.h"
#include "py_optimizer_def.h"

#include "../macros.h"
using Optimizer = sail::optimizers::Optimizer;

static int PyOptimizer_init(PyOptimizer *self, PyObject *args,
                            PyObject *kwargs) {
    self->optimizer = new Optimizer();
    return 0;
}
static int PyOptimizer_traverse(PyOptimizer *self, visitproc visit, void *arg) {
    Py_VISIT(self->base_object);
    return 0;
}

static int PyOptimizer_clear(PyOptimizer *self) {
    delete self->optimizer;
    if (self->base_object != NULL) {
        Py_DECREF(self->base_object);
    }
    return 0;
}

static void PyOptimizer_dealloc(PyOptimizer *self) {
    PyObject_GC_UnTrack(self);
    PyOptimizer_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

RETURN_OBJECT
PyOptimizer_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyOptimizer *self;
    self = (PyOptimizer *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/////////////////////////////////////////

RETURN_OBJECT PyOptimizer_track_module(PyOptimizer *self, PyObject *args,
                                       PyObject *kwargs) {
    //     PyErr_SetString(PyExc_NotImplementedError, "");
    //     return NULL;
    PyModule *mod = nullptr;
    static char *kwlist[] = {"module", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &mod)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }
    self->optimizer->track_module(*(mod->module));
    Py_INCREF(mod);
    Py_RETURN_NONE;
}

RETURN_OBJECT PyOptimizer_update(PyOptimizer *self) {
    //     PyErr_SetString(PyExc_NotImplementedError, "");
    //     return NULL;

    self->optimizer->update();
    Py_RETURN_NONE;
}
