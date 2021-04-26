/*
################################################################################
#                  THIS CODE IS AUTOGENERATED FROM A TEMPLATE                  #
#                 TO MAKE CHANGES, EDIT THE ORIGINAL .src FILE                 #
################################################################################
*/

#pragma once

#include <Python.h>
#include "numpy/arrayobject.h"
#include "py_tensor.h"
#include <structmember.h>
#include "../../src/Tensor.h"
#include "../../src/error.h"
#include "../../src/dtypes.h"
#include <chrono>
#include <iostream>
#include "../macros.h"

RETURN_OBJECT PyDimensionError;

/** begin block
 * name = [add, sub, mul, truediv]
 * op = [+, -, *, /]
 */



RETURN_OBJECT PyTensor_add(PyObject *self, PyObject *other) {

    BINARY_TENSOR_TYPE_CHECK(self, other);

    PyTensor* ret_class;
    ret_class = (PyTensor *) PyTensorType.tp_alloc(&PyTensorType, 0);

    try {
        ret_class->tensor = ((PyTensor*)self)->tensor + ((PyTensor*)other)->tensor;
    } catch (DimensionError& err) {
        PyObject *tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyUnicode_FromString(err.what()));
        PyTuple_SetItem(tuple, 1, PyLong_FromLong(257));
        PyErr_SetObject(PyDimensionError, tuple);
        return NULL;
    }

    ret_class->ndim = ((PyTensor*)self)->ndim;
    ret_class->dtype = ((PyTensor*)self)->dtype;

    return (PyObject *) ret_class;
}




RETURN_OBJECT PyTensor_sub(PyObject *self, PyObject *other) {

    BINARY_TENSOR_TYPE_CHECK(self, other);

    PyTensor* ret_class;
    ret_class = (PyTensor *) PyTensorType.tp_alloc(&PyTensorType, 0);

    try {
        ret_class->tensor = ((PyTensor*)self)->tensor - ((PyTensor*)other)->tensor;
    } catch (DimensionError& err) {
        PyObject *tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyUnicode_FromString(err.what()));
        PyTuple_SetItem(tuple, 1, PyLong_FromLong(257));
        PyErr_SetObject(PyDimensionError, tuple);
        return NULL;
    }

    ret_class->ndim = ((PyTensor*)self)->ndim;
    ret_class->dtype = ((PyTensor*)self)->dtype;

    return (PyObject *) ret_class;
}




RETURN_OBJECT PyTensor_mul(PyObject *self, PyObject *other) {

    BINARY_TENSOR_TYPE_CHECK(self, other);

    PyTensor* ret_class;
    ret_class = (PyTensor *) PyTensorType.tp_alloc(&PyTensorType, 0);

    try {
        ret_class->tensor = ((PyTensor*)self)->tensor * ((PyTensor*)other)->tensor;
    } catch (DimensionError& err) {
        PyObject *tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyUnicode_FromString(err.what()));
        PyTuple_SetItem(tuple, 1, PyLong_FromLong(257));
        PyErr_SetObject(PyDimensionError, tuple);
        return NULL;
    }

    ret_class->ndim = ((PyTensor*)self)->ndim;
    ret_class->dtype = ((PyTensor*)self)->dtype;

    return (PyObject *) ret_class;
}




RETURN_OBJECT PyTensor_truediv(PyObject *self, PyObject *other) {

    BINARY_TENSOR_TYPE_CHECK(self, other);

    PyTensor* ret_class;
    ret_class = (PyTensor *) PyTensorType.tp_alloc(&PyTensorType, 0);

    try {
        ret_class->tensor = ((PyTensor*)self)->tensor / ((PyTensor*)other)->tensor;
    } catch (DimensionError& err) {
        PyObject *tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyUnicode_FromString(err.what()));
        PyTuple_SetItem(tuple, 1, PyLong_FromLong(257));
        PyErr_SetObject(PyDimensionError, tuple);
        return NULL;
    }

    ret_class->ndim = ((PyTensor*)self)->ndim;
    ret_class->dtype = ((PyTensor*)self)->dtype;

    return (PyObject *) ret_class;
}

/** end block **/
