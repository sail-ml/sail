#pragma once

#include <Python.h>
#include <core/Tensor.h>
#include <core/ops/ops.h>
#include <core/tensor_shape.h>
#include <core/types.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include "../../py_tensor/py_tensor.h"
#include "numpy/arrayobject.h"

#include "../../error_defs.h"
#include "../../macros.h"

RETURN_OBJECT ops_tensordot(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    PyObject* t1;
    PyObject* t2;
    PyObject* tuple = Py_None;
    int v = 2;

    sail::Tensor tensor1;
    sail::Tensor tensor2;

    static char* kwlist[] = {"a", "b", "axes", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &t1, &t2,
                                     &tuple)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    tensor1 = ((PyTensor*)t1)->tensor;
    tensor2 = ((PyTensor*)t2)->tensor;

    std::vector<long> axes_1;
    std::vector<long> axes_2;

    if (PyTuple_Check(tuple)) {
        PyObject* tuple1 = PyTuple_GetItem(tuple, 0);
        PyObject* tuple2 = PyTuple_GetItem(tuple, 1);

        tuple1 = PySequence_Tuple(tuple1);
        tuple2 = PySequence_Tuple(tuple2);

        int len = PyTuple_Size(tuple1);
        if (len == -1) {
            axes_1 = {PyLong_AsLong(tuple1)};
        } else {
            while (len--) {
                axes_1.push_back(PyLong_AsLong(PyTuple_GetItem(tuple1, len)));
            }
            std::reverse(axes_1.begin(), axes_1.end());
        }
        len = PyTuple_Size(tuple2);
        if (len == -1) {
            axes_2 = {PyLong_AsLong(tuple2)};
        } else {
            while (len--) {
                axes_2.push_back(PyLong_AsLong(PyTuple_GetItem(tuple2, len)));
            }
            std::reverse(axes_2.begin(), axes_2.end());
        }
    } else {
        if (PyLong_Check(tuple)) {
            v = PyLong_AsLong(tuple);
        }
        for (int i = tensor1.get_ndim() - 1; i > (tensor1.get_ndim() - v - 1);
             i--) {
            axes_1.insert(axes_1.begin(), i);
        }

        for (int i = 0; i < v; i++) {
            axes_2.push_back(i);
        }
    }

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    sail::Tensor res = sail::ops::tensordot(tensor1, tensor2, axes_1, axes_2);

    ret_class->tensor = res;

    return (PyObject*)ret_class;
    END_EXCEPTION_HANDLING
}

// RETURN_OBJECT ops_matmul(PyObject* self, PyObject* args) {
//     START_EXCEPTION_HANDLING

//     PyObject* t1;
//     PyObject* t2;

//     sail::Tensor tensor1;
//     sail::Tensor tensor2;

//     if (!PyArg_ParseTuple(args, "OO", &t1, &t2)) {
//         PyErr_SetString(PyExc_TypeError, "Inputs should be Sail Tensors");
//         return nullptr;
//     }

//     tensor1 = ((PyTensor*)t1)->tensor;
//     tensor2 = ((PyTensor*)t2)->tensor;

//     PyTensor* ret_class;
//     ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

//     sail::Tensor res = sail::ops::matmul(tensor1, tensor2);

//     ret_class->tensor = res;

//     return (PyObject*)ret_class;
//     END_EXCEPTION_HANDLING
// }

// RETURN_OBJECT ops_addmm(PyObject* self, PyObject* args) {
//     START_EXCEPTION_HANDLING
//     PyObject* t1;
//     PyObject* t2;
//     PyObject* t3;

//     sail::Tensor tensor1;
//     sail::Tensor tensor2;
//     sail::Tensor tensor3;

//     if (!PyArg_ParseTuple(args, "OOO", &t1, &t2, &t3)) {
//         PyErr_SetString(PyExc_TypeError, "Incorrect arguments");
//         return nullptr;
//     }

//     tensor1 = ((PyTensor*)t1)->tensor;
//     tensor2 = ((PyTensor*)t2)->tensor;
//     tensor3 = ((PyTensor*)t3)->tensor;

//     PyTensor* ret_class;
//     ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

//     sail::Tensor res = sail::ops::addmm(tensor1, tensor2, tensor3);

//     ret_class->tensor = res;

//     return (PyObject*)ret_class;
//     END_EXCEPTION_HANDLING
// }
