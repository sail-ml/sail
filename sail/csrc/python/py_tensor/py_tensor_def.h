#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/dtypes.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

using SCTensor = sail::Tensor;

typedef struct {
    PyObject_HEAD PyObject *base_object = NULL;
    SCTensor tensor;
    int ndim;
    int dtype;
    bool requires_grad = false;
} PyTensor;

///////////////////// DEFINITIONS ///////////////////////

/////////////// BASE PY FCNS ////////////////
static int PyTensor_init(PyTensor *self, PyObject *args, PyObject *kwargs);
static int PyTensor_traverse(PyTensor *self, visitproc visit, void *arg);
static int PyTensor_clear(PyTensor *self);
static void PyTensor_dealloc(PyTensor *self);
RETURN_OBJECT PyTensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
RETURN_OBJECT PyTensor_repr(PyTensor *self);

/////////////// ARITHMETIC /////////////////
RETURN_OBJECT PyTensor_add(PyObject *self, PyObject *other);
RETURN_OBJECT PyTensor_sub(PyObject *self, PyObject *other);
RETURN_OBJECT PyTensor_mul(PyObject *self, PyObject *other);
RETURN_OBJECT PyTensor_truediv(PyObject *self, PyObject *other);

///////////// MAPPING ///////////////////
RETURN_OBJECT PyTensor_getitem(PyObject *self, PyObject *key);

///////////// GET SET //////////////////
RETURN_OBJECT PyTensor_get_shape(PyTensor *self, void *closure);
static int PyTensor_set_shape(PyTensor *self,
                              void *closure);  // DOES NOTHING

//////////// CLASS METHODS ////////////////
RETURN_OBJECT PyTensor_get_ndim(PyTensor *self, void *closure);
RETURN_OBJECT PyTensor_get_numpy(PyTensor *self, void *closure);
RETURN_OBJECT PyTensor_astype(PyObject *self, PyObject *args, void *closure);
RETURN_OBJECT PyTensor_backward(PyTensor *self, void *closure);
RETURN_OBJECT PyTensor_get_grad(PyTensor *self, void *closure);

//////////// DEF ARRAYS ///////////////////
static PyMemberDef PyTensor_members[] = {
    {"ndim", T_INT, offsetof(PyTensor, ndim), 0, "dimensions"},
    {"requires_grad", T_BOOL, offsetof(PyTensor, requires_grad), 0,
     "requires_grad"},
    {NULL}};

static PyMethodDef PyTensor_methods[] = {
    {"numpy", (PyCFunction)PyTensor_get_numpy, METH_VARARGS,
     "Return the name, combining the first and last name"},
    {"astype", (PyCFunction)PyTensor_astype, METH_VARARGS, "Casts the tensor"},
    {"get_grad", (PyCFunction)PyTensor_get_grad, METH_VARARGS,
     "PyTensor_get_grad"},
    {"backward", (PyCFunction)PyTensor_backward, METH_VARARGS,
     "PyTensor_backward"},
    {NULL} /* Sentinel */
};

static PyGetSetDef PyTensor_get_setters[] = {
    {"shape", (getter)PyTensor_get_shape, (setter)PyTensor_set_shape, "shape",
     NULL},
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
    (binaryfunc)PyTensor_sub,  // nb_sub,
    (binaryfunc)PyTensor_mul,  // nb_mul,
    0,                         // nb_remainder
    0,                         // nb_divmod
    0,                         // nb_power
    0,                         // nb_negative
    0,                         // nb_positive
    0,                         // nb_absolute
    0,                         // nb_bool
    0,                         // nb_invert
    0,                         // nb_lshift
    0,                         // nb_rshift
    0,                         // nb_and
    0,                         // nb_xor
    0,                         // nb_or
    0,                         // nb_int
    0,                         // nb_reserved
    0,                         // nb_float

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

    0,                             // mp_length
    (binaryfunc)PyTensor_getitem,  // mp_subscript
    0,                             // mp_ass_subscript
};

//////////////// TYPE DEF ////////////////////
static PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0) "libsail_c.Tensor", /* tp_name */
    sizeof(PyTensor),                                  /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)PyTensor_dealloc,                      /* tp_dealloc */
    0,                                                 /* tp_print */
    0,                                                 /* tp_getattr */
    0,                                                 /* tp_setattr */
    0,                                                 /* tp_reserved */
    (reprfunc)PyTensor_repr,                           /* tp_repr */
    &PyTensorNumberMethods,                            /* tp_as_number */
    0,                                                 /* tp_as_sequence */
    &PyTensorMappingMethods,                           /* tp_as_mapping */
    0,                                                 /* tp_hash */
    0,                                                 /* tp_call */
    0,                                                 /* tp_str */
    0,                                                 /* tp_getattro */
    0,                                                 /* tp_setattro */
    0,                                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,          /* tp_flags */
    "Custom objects",                /* tp_doc */
    (traverseproc)PyTensor_traverse, /* tp_traverse */
    (inquiry)PyTensor_clear,         /* tp_clear */
    0,                               /* tp_richcompare */
    0,                               /* tp_weaklistoffset */
    0,                               /* tp_iter */
    0,                               /* tp_iternext */
    PyTensor_methods,                /* tp_methods */
    PyTensor_members,                /* tp_members */
    PyTensor_get_setters,            // PyTensor_getsetters, /* tp_getset */
    0,                               /* tp_base */
    0,                               /* tp_dict */
    0,                               /* tp_descr_get */
    0,                               /* tp_descr_set */
    0,                               /* tp_dictoffset */
    (initproc)PyTensor_init,         /* tp_init */
    0,                               /* tp_alloc */
    PyTensor_new                     /* tp_new */

};