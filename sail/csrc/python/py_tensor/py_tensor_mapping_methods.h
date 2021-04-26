#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/dtypes.h"
#include "numpy/arrayobject.h"
#include "py_tensor.h"

#include "../macros.h"

RETURN_OBJECT PyTensor_getitem(PyObject *self, PyObject *key) {
    int idx = static_cast<int>(PyLong_AsLong(key));

    if (idx > ((PyTensor *)self)->tensor.storage.numel()) {
        return NULL;
    }

    PyTensor *ret_class;
    ret_class = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = ((PyTensor *)self)->tensor[idx];

    // ret_class->base_object = self;
    // Py_INCREF(self);

    ret_class->ndim = ((PyTensor *)ret_class)->tensor.storage.ndim;
    ret_class->dtype = ((PyTensor *)self)->dtype;

    return (PyObject *)ret_class;
}
