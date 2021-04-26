#pragma once 

#include <Python.h>
#include "numpy/arrayobject.h"
#include <structmember.h>
#include "../../src/Tensor.h"
#include "../../src/dtypes.h"
#include <chrono>
#include <iostream>

#include "../macros.h"
// using RETURN_OBJECT = RETURN_OBJECT;

static int
PyTensor_init(PyTensor *self, PyObject *args)
{
    PyArrayObject *array;
    // Py_INCREF(args);
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        PyErr_SetString(PyExc_TypeError, "Input should be a numpy array of numbers.");
        return NULL;
    }
  
    int ndim = PyArray_NDIM(array);
    int dtype = PyArray_TYPE(array);

    void* data = std::move(static_cast<void*>(array->data));
    Dtype dt = GetDtypeFromNumpyInt(dtype);
    std::vector<py::ssize_t> shape, strides;

    long int* shape_ptr = PyArray_SHAPE(array);
    long int* stride_ptr = PyArray_STRIDES(array);

    for (int i = 0; i < ndim; i ++) {
        shape.push_back(shape_ptr[i]);
        strides.push_back(stride_ptr[i]);
    }

    SCTensor tensor = SCTensor(ndim, data, dt, strides, shape);
    self->tensor = tensor;

    // Py_INCREF(ndim);
    // Py_INCREF(typ);
    // Py_INCREF(data);

    self->ndim = ndim;
    self->dtype = dtype;

    return 0;
}
static int
PyTensor_traverse(PyTensor *self, visitproc visit, void *arg)
{
    Py_VISIT(self->base_object);
    // Py_VISIT(self->last);
    return 0;
}

static int
PyTensor_clear(PyTensor *self)
{
    self->tensor.free();
    return 0;
}

static void
PyTensor_dealloc(PyTensor *self)
{
    PyObject_GC_UnTrack(self);
    PyTensor_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

RETURN_OBJECT
PyTensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyTensor *self;
    self = (PyTensor *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ndim = 0;
    }
    // self = (CustomObject *) type->tp_alloc(type, 0);
    // if (self != NULL) {
    //     self->first = PyUnicode_FromString("");
    //     if (self->first == NULL) {
    //         Py_DECREF(self);
    //         return NULL;
    //     }
    //     self->last = PyUnicode_FromString("");
    //     if (self->last == NULL) {
    //         Py_DECREF(self);
    //         return NULL;
    //     }
    //     self->number = 0;
    // }
    return (PyObject *) self;
}


RETURN_OBJECT
PyTensor_get_ndim(PyTensor *self, void *closure)
{
    Py_INCREF(self->ndim);
    long x = static_cast<long>(self->ndim);
    return PyLong_FromLong(x);
}

RETURN_OBJECT
PyTensor_get_numpy(PyTensor *self, void* closure) {
    int ndims = self->ndim;
    long int* shape = self->tensor.getShapePtr();

    int type = self->tensor.getNPTypeNum();
    void* data = std::move(static_cast<void*>(self->tensor.storage.data));
    Py_INCREF(self);

    PyObject* array = PyArray_SimpleNewFromData(ndims, shape, type, data);
    PyArray_SetBaseObject((PyArrayObject *) array, (PyObject *) self);
    return PyArray_Return((PyArrayObject *) array);
}
