#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../py_module/py_module.h"
#include "core/Tensor.h"
#include "core/TensorBody.h"
#include "core/dtypes.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "core/types.h"
#include "numpy/arrayobject.h"
#include "py_optimizer_def.h"

#include "../error_defs.h"
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
    START_EXCEPTION_HANDLING
    //     PyErr_SetString(PyExc_NotImplementedError, "");
    //     return nullptr;
    PyModule *mod = nullptr;
    static char *kwlist[] = {"module", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &mod)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }
    std::cout << "ja" << std::endl;
    self->optimizer->track_module(*(mod->module));
    Py_INCREF(mod);
    Py_RETURN_NONE;
    END_EXCEPTION_HANDLING
}

RETURN_OBJECT PyOptimizer_update(PyOptimizer *self) {
    START_EXCEPTION_HANDLING
    //     PyErr_SetString(PyExc_NotImplementedError, "");
    //     return nullptr;

    self->optimizer->update();
    Py_RETURN_NONE;
    END_EXCEPTION_HANDLING
}
