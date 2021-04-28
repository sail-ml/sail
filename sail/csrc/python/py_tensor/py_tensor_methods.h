#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/dtypes.h"
#include "../../src/types.h"
#include "../py_dtypes/py_dtype.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

#define CAST_TYPE_CHECK(args, x)                               \
    {                                                          \
        if (!PyArg_ParseTuple(args, "O!", &PyDtypeBase, &x)) { \
            return NULL;                                       \
        }                                                      \
    }
// #define CAST_TYPE_CHECK(x)                                   \
//     {                                                        \
//         if (PyObject_TypeCheck(x, &PyDtypeInt32)) {          \
//         } else if (PyObject_TypeCheck(x, &PyDtypeFloat32)) { \
//         } else if (PyObject_TypeCheck(x, &PyDtypeFloat64)) { \
//         } else {                                             \
//             return NULL;                                     \
//         }                                                    \
//     }

// using RETURN_OBJECT = RETURN_OBJECT;

static int PyTensor_init(PyTensor *self, PyObject *args) {
    PyArrayObject *array;
    // Py_INCREF(args);
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        PyErr_SetString(PyExc_TypeError,
                        "Input should be a numpy array of numbers.");
        return NULL;
    }

    int ndim = PyArray_NDIM(array);
    int dtype = PyArray_TYPE(array);

    void *data = std::move(static_cast<void *>(array->data));
    Dtype dt = GetDtypeFromNumpyInt(dtype);
    TensorShape shape, strides;

    long int *shape_ptr = PyArray_SHAPE(array);
    long int *stride_ptr = PyArray_STRIDES(array);

    for (int i = 0; i < ndim; i++) {
        shape.push_back(shape_ptr[i]);
        strides.push_back(stride_ptr[i]);
    }

    SCTensor tensor = SCTensor(ndim, data, dt, strides, shape);
    self->tensor = tensor;

    self->ndim = ndim;
    self->dtype = dtype;

    return 0;
}
static int PyTensor_traverse(PyTensor *self, visitproc visit, void *arg) {
    Py_VISIT(self->base_object);
    // Py_VISIT(self->last);
    return 0;
}

static int PyTensor_clear(PyTensor *self) {
    self->tensor.free();
    return 0;
}

static void PyTensor_dealloc(PyTensor *self) {
    PyObject_GC_UnTrack(self);
    PyTensor_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

RETURN_OBJECT
PyTensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyTensor *self;
    self = (PyTensor *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ndim = 0;
    }

    return (PyObject *)self;
}

RETURN_OBJECT
PyTensor_get_ndim(PyTensor *self, void *closure) {
    Py_INCREF(self->ndim);
    long x = static_cast<long>(self->ndim);
    return PyLong_FromLong(x);
}

RETURN_OBJECT
PyTensor_get_numpy(PyTensor *self, void *closure) {
    int ndims = self->ndim;
    long int *shape = self->tensor.get_shape_ptr();

    int type = self->tensor.get_np_type_num();
    void *data = std::move(static_cast<void *>(self->tensor.data));
    Py_INCREF(self);

    PyObject *array = PyArray_SimpleNewFromData(ndims, shape, type, data);
    PyArray_SetBaseObject((PyArrayObject *)array, (PyObject *)self);
    return PyArray_Return((PyArrayObject *)array);
}

RETURN_OBJECT
PyTensor_astype(PyObject *self, PyObject *args, void *closure) {
    PyDtype *type;

    CAST_TYPE_CHECK(args, type);

    Dtype dt = ((PyDtype *)type)->dtype;

    PyTensor *ret_class;
    ret_class = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = ((PyTensor *)self)->tensor.cast(Dtype::sInt32);

    ret_class->ndim = ret_class->tensor.ndim;
    ret_class->dtype = ((PyDtype *)type)->dt_val;

    return (PyObject *)ret_class;
}
