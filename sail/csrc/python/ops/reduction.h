#pragma once 

#include <Python.h>
#include "numpy/arrayobject.h"
#include <structmember.h>
#include "../../src/Tensor.h"
#include "../../src/ops/reduction.h"
#include "../py_tensor/py_tensor.h"
#include <chrono>
#include <iostream>

#include "../macros.h"

/** begin block
 * name = [add, sub, mul, div]
 * op = [+, -, *, /]
 */

RETURN_OBJECT ops_sum(PyObject* self, PyObject *args) {

    PyTensor* t1;

    if (!PyArg_ParseTuple(args, "O", &t1)) {
        PyErr_SetString(PyExc_TypeError, "Inputs should be Sail Tensors");
        return NULL;
    }

    PyTensor* ret_class;
    ret_class = (PyTensor *) PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = sail::ops::sum(((PyTensor*)t1)->tensor);

    ret_class->ndim = ret_class->tensor.storage.ndim;
    ret_class->dtype = ((PyTensor*)t1)->dtype;

    return (PyObject *) ret_class;
}

/** end block **/


