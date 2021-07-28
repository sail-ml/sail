// allow-no-source allow-comments
#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "numpy/arrayobject.h"

using SCTensor = sail::Tensor;

typedef struct {
    PyObject_HEAD;
    PyObject *base_object = NULL;
    SCTensor tensor;
} PyTensor;

///////////////////// DEFINITIONS ///////////////////////

/////////////// BASE PY FCNS ////////////////
static int PyTensor_init(PyTensor *self, PyObject *args, PyObject *kwargs);
static int PyTensor_traverse(PyTensor *self, visitproc visit, void *arg);
static int PyTensor_clear(PyTensor *self);
static void PyTensor_dealloc(PyTensor *self);
static PyObject *PyTensor_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwds);
static PyObject *PyTensor_repr(PyTensor *self);
static PyObject *PyTensor_RichCompare(PyObject *self, PyObject *other, int op);

/////////////// ARITHMETIC /////////////////
static PyObject *PyTensor_add(PyObject *self, PyObject *other);
static PyObject *PyTensor_sub(PyObject *self, PyObject *other);
static PyObject *PyTensor_mul(PyObject *self, PyObject *other);
static PyObject *PyTensor_truediv(PyObject *self, PyObject *other);
static PyObject *PyTensor_negate(PyObject *self);

///////////// MAPPING ///////////////////
static PyObject *PyTensor_getitem(PyObject *self, PyObject *key);

///////////// GET SET //////////////////
static PyObject *PyTensor_get_shape(PyTensor *self, void *closure);
static int PyTensor_set_shape(PyTensor *self,
                              void *closure);  // DOES NOTHING
static PyObject *PyTensor_get_grad(PyTensor *self, void *closure);
static int PyTensor_set_grad(PyTensor *self, void *closure);
static PyObject *PyTensor_get_requires_grad(PyTensor *self, void *closure);
static int PyTensor_set_requires_grad(PyTensor *self, PyObject *value,
                                      void *closure);
static PyObject *PyTensor_get_ndim(PyTensor *self, void *closure);
static int PyTensor_set_ndim(PyTensor *self, PyObject *value, void *closure);

//////////// CLASS METHODS ////////////////
static PyObject *PyTensor_get_numpy(PyTensor *self, void *closure);
static PyObject *PyTensor_astype(PyObject *self, PyObject *args, void *closure);
static PyObject *PyTensor_backward(PyTensor *self, void *closure);
static long PyTensor_len(PyTensor *self);

//////////// DEF ARRAYS ///////////////////

static PyMethodDef PyTensor_methods[] = {
    {"numpy", (PyCFunction)PyTensor_get_numpy, METH_VARARGS},
    {"astype", (PyCFunction)PyTensor_astype, METH_VARARGS, NULL},
    {"get_grad", (PyCFunction)PyTensor_get_grad, METH_VARARGS, NULL},
    {"backward", (PyCFunction)PyTensor_backward, METH_VARARGS, NULL},
    {NULL} /* Sentinel */
};

static PyGetSetDef PyTensor_get_setters[] = {
    {"shape", (getter)PyTensor_get_shape, (setter)PyTensor_set_shape, NULL},
    {"ndim", (getter)PyTensor_get_ndim, (setter)PyTensor_set_ndim, NULL},
    {"grad", (getter)PyTensor_get_grad, (setter)PyTensor_set_grad, NULL},
    {"requires_grad", (getter)PyTensor_get_requires_grad,
     (setter)PyTensor_set_requires_grad, NULL},
    {NULL} /* Sentinel */
};

////////////// NUMBER METHODS //////////////////////
static PyNumberMethods PyTensorNumberMethods = {
    /* PyNumberMethods, implementing the number protocol
     * references:
     * https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods
     * https://docs.python.org/3/c-api/number.html
     */
    (binaryfunc)PyTensor_add,
    (binaryfunc)PyTensor_sub,    // nb_sub,
    (binaryfunc)PyTensor_mul,    // nb_mul,
    0,                           // nb_remainder
    0,                           // nb_divmod
    0,                           // nb_power
    (unaryfunc)PyTensor_negate,  // nb_negative
    0,                           // nb_positive
    0,                           // nb_absolute
    0,                           // nb_bool
    0,                           // nb_invert
    0,                           // nb_lshift
    0,                           // nb_rshift
    0,                           // nb_and
    0,                           // nb_xor
    0,                           // nb_or
    0,                           // nb_int
    0,                           // nb_reserved
    0,                           // nb_float

    0,  // nb_inplace_add
    0,  // nb_inplace_subtract
    0,  // nb_inplace_multiply
    0,  // nb_inplace_remainder
    0,  // nb_inplace_power
    0,  // nb_inplace_lshift
    0,  // nb_inplace_rshift
    0,  // nb_inplace_and
    0,  // nb_inplace_xor
    0,  // nb_inplace_or

    0,                             // nb_floor_divide
    (binaryfunc)PyTensor_truediv,  // nb_true_divide
    0,                             // nb_inplace_floor_divide
    0,                             // nb_inplace_true_divide

    0,  // nb_index
};

////////////// MAPPING METHODS //////////////
static PyMappingMethods PyTensorMappingMethods = {
    // https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods

    (lenfunc)PyTensor_len,         // mp_length
    (binaryfunc)PyTensor_getitem,  // mp_subscript
    0,                             // mp_ass_subscript
};

//////////////// TYPE DEF ////////////////////
static PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0) "sail.Tensor", /* tp_name */
    sizeof(PyTensor),                             /* tp_basicsize */
    0,                                            /* tp_itemsize */
    (destructor)PyTensor_dealloc,                 /* tp_dealloc */
    0,                                            /* tp_print */
    0,                                            /* tp_getattr */
    0,                                            /* tp_setattr */
    0,                                            /* tp_reserved */
    (reprfunc)PyTensor_repr,                      /* tp_repr */
    &PyTensorNumberMethods,                       /* tp_as_number */
    0,                                            /* tp_as_sequence */
    &PyTensorMappingMethods,                      /* tp_as_mapping */
    0,                                            /* tp_hash */
    0,                                            /* tp_call */
    0,                                            /* tp_str */
    0,                                            /* tp_getattro */
    0,                                            /* tp_setattro */
    0,                                            /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,            /* tp_flags */
    "Custom objects",                  /* tp_doc */
    (traverseproc)PyTensor_traverse,   /* tp_traverse */
    (inquiry)PyTensor_clear,           /* tp_clear */
    (richcmpfunc)PyTensor_RichCompare, /* tp_richcompare */
    0,                                 /* tp_weaklistoffset */
    0,                                 /* tp_iter */
    0,                                 /* tp_iternext */
    PyTensor_methods,                  /* tp_methods */
    0,                                 /* tp_members */
    PyTensor_get_setters,              // PyTensor_getsetters, /* tp_getset */
    0,                                 /* tp_base */
    0,                                 /* tp_dict */
    0,                                 /* tp_descr_get */
    0,                                 /* tp_descr_set */
    0,                                 /* tp_dictoffset */
    (initproc)PyTensor_init,           /* tp_init */
    0,                                 /* tp_alloc */
    PyTensor_new,                      /* tp_new */
    PyObject_GC_Del                    /* tp_free */

};
