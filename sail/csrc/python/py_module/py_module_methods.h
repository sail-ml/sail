#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/TensorBody.h"
#include "core/dtypes.h"
#include "core/factories.h"
#include "core/modules/module.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "core/types.h"
#include "numpy/arrayobject.h"
#include "py_module_def.h"

#include "../macros.h"

static int PyModule_init(PyModule *self, PyObject *args, PyObject *kwargs) {
    self->module = new sail::modules::Module();
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
    Py_RETURN_NONE;
}
int PyModule_setattr(PyModule *self, PyObject *attr, PyObject *value) {
    
    if (PyObject_IsInstance(value, (PyObject *)&PyModuleType)) {
        self->module->register_params(((PyModule *)value)->module->params);
    } 
    PyObject_GenericSetAttr((PyObject*)self, attr, value);
    
    return 0;
}
