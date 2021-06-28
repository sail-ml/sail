#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "numpy/arrayobject.h"
#include "py_tensor.h"

#include "../error_defs.h"
#include "../macros.h"

static PyObject *PyTensor_getitem(PyObject *self, PyObject *key) {
    START_EXCEPTION_HANDLING
    PyTensor *ret_class;
    ret_class = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (PySlice_Check(key)) {
        long start, stop, step;
        PySlice_GetIndices(key, ((PyTensor *)self)->tensor.len(), &start, &stop,
                           &step);
        ret_class->tensor = ((PyTensor *)self)->tensor.slice(start, stop);
    } else {
        int idx = static_cast<int>(PyLong_AsLong(key));

        if (idx > ((PyTensor *)self)->tensor.numel()) {
            return nullptr;
        }

        ret_class->tensor = ((PyTensor *)self)->tensor[idx];
    }

    SET_BASE(self, ret_class);

    return (PyObject *)ret_class;
    END_EXCEPTION_HANDLING
}
