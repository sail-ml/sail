#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/dtypes.h"
#include "../../src/tensor_shape.h"
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

static int PyTensor_init(PyTensor *self, PyObject *args, PyObject *kwargs) {
    PyArrayObject *array;
    // Py_INCREF(args);
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        PyErr_SetString(PyExc_TypeError,
                        "Input should be a numpy array of numbers.");
        return NULL;
    }

    bool requires_grad = false;
    static char *kwlist[] = {"array", "requires_grad", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|b", kwlist, &array,
                                     &requires_grad)) {
        PyErr_SetString(PyExc_TypeError,
                        "must pass a tensor and a bool for requires_grad");
    }

    int ndim = PyArray_NDIM(array);
    int dtype = PyArray_TYPE(array);

    void *data = std::move(static_cast<void *>(array->data));
    Dtype dt = GetDtypeFromNumpyInt(dtype);
    TensorSize shape, strides;

    long int *shape_ptr = PyArray_SHAPE(array);
    // 0 check, cant have an array that is size 0
    long int *stride_ptr = PyArray_STRIDES(array);

    for (int i = 0; i < ndim; i++) {
        shape.push_back(shape_ptr[i]);
        strides.push_back(stride_ptr[i]);
    }

    SCTensor tensor = SCTensor(
        ndim, data, dt, sail::TensorShape(shape, strides), requires_grad);
    self->tensor = tensor;

    self->ndim = ndim;
    self->requires_grad = requires_grad;
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
PyObject *inner_numpy(sail::Tensor tensor) {
    int ndims = tensor.get_ndim();
    long int *shape = tensor.get_shape_ptr();

    int type = tensor.get_np_type_num();
    void *data = malloc(tensor.getTotalSize());  // self->tensor.data;

    memcpy(data, tensor.data, tensor.getTotalSize());
    PyObject *array;
    if (!tensor.broadcasted) {
        array = PyArray_SimpleNewFromData(ndims, shape, type, data);
    } else {
        PyObject *temp_array;
        shape = tensor.old_shape.get_shape_ptr();
        ndims = tensor.old_shape.ndim();
        array = PyArray_SimpleNewFromData(ndims, shape, type, data);

        PyArray_BroadcastToShape(array, tensor.get_shape_ptr(),
                                 tensor.get_ndim());
    }
    return array;
}
RETURN_OBJECT
PyTensor_get_numpy(PyTensor *self, void *closure) {
    Py_INCREF(self);
    PyObject *array = inner_numpy(self->tensor);
    std::cout << "returning again" << std::endl;

    // PyArray_SetBaseObject((PyArrayObject *)array, (PyObject *)self);
    return PyArray_Return((PyArrayObject *)array);
}

RETURN_OBJECT PyTensor_get_grad(PyTensor *self, void *closure) {
    Py_INCREF(self);
    PyObject *array = inner_numpy(*(self->tensor.grad));
    // PyArray_SetBaseObject((PyArrayObject *)array, (PyObject *)self);
    return PyArray_Return((PyArrayObject *)array);
}

RETURN_OBJECT
PyTensor_astype(PyObject *self, PyObject *args, void *closure) {
    PyDtype *type;

    CAST_TYPE_CHECK(args, type);

    Dtype dt = ((PyDtype *)type)->dtype;

    PyTensor *ret_class;
    ret_class = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = ((PyTensor *)self)->tensor.cast(dt);

    ret_class->ndim = ret_class->tensor.ndim;
    ret_class->dtype = ((PyDtype *)type)->dt_val;

    return (PyObject *)ret_class;
}

RETURN_OBJECT
PyTensor_get_shape(PyTensor *self, void *closure) {
    PyObject *tuple = PyTuple_New(self->tensor.get_ndim());
    int c = 0;
    for (long s : self->tensor.shape_details.shape) {
        PyTuple_SetItem(tuple, c, PyLong_FromLong(s));
        c += 1;
    }
    return tuple;
}
static int PyTensor_set_shape(PyTensor *self, void *closure) {
    PyErr_SetString(PyExc_AttributeError,
                    "Shape cannot be modified like this. Use reshape");
    return -1;
}

RETURN_OBJECT PyTensor_backward(PyTensor *self, void *closure) {
    self->tensor.backward();
    return (PyObject *)self;
}
