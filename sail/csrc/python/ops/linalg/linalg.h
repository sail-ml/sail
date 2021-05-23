#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include "../../../src/Tensor.h"
#include "../../../src/ops/ops.h"
#include "../../../src/tensor_shape.h"
#include "../../../src/types.h"
#include "../../py_tensor/py_tensor.h"
#include "numpy/arrayobject.h"

#include "../../macros.h"

RETURN_OBJECT ops_tensordot(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* t1;
    PyObject* t2;
    PyObject* tuple = Py_None;
    int v = 2;

    sail::Tensor tensor1;
    sail::Tensor tensor2;

    static char* kwlist[] = {"a", "b", "axes", NULL};

    std::cout << "ici" << std::endl;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &t1, &t2,
                                     &tuple)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }
    std::cout << "aci" << std::endl;

    tensor1 = ((PyTensor*)t1)->tensor;
    std::cout << "aci" << std::endl;
    tensor2 = ((PyTensor*)t2)->tensor;
    std::cout << "aci" << std::endl;

    std::vector<long> axes_1;
    std::vector<long> axes_2;

    std::cout << "moving in" << std::endl;

    if (PyTuple_Check(tuple)) {
        PyObject* tuple1 = PyTuple_GetItem(tuple, 0);
        PyObject* tuple2 = PyTuple_GetItem(tuple, 1);
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
            std::cout << "ja" << std::endl;
        }
    } else {
        if (PyLong_Check(tuple)) {
            v = PyLong_AsLong(tuple);
        }
        for (int i = tensor1.get_ndim() - 1; i > (v - tensor1.get_ndim() + 1);
             i--) {
            axes_1.insert(axes_1.begin(), i);
        }

        for (int i = 0; i < v; i++) {
            axes_2.push_back(i);
        }
    }

    std::cout << getVectorString(axes_1) << std::endl;
    std::cout << getVectorString(axes_2) << std::endl;

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    sail::Tensor res = sail::ops::tensordot(tensor1, tensor2, axes_1, axes_2);

    ret_class->tensor = res;
    ret_class->ndim = ((PyTensor*)t1)->ndim;
    ret_class->dtype = ((PyTensor*)t1)->dtype;

    return (PyObject*)ret_class;
}

RETURN_OBJECT ops_matmul(PyObject* self, PyObject* args) {
    PyObject* t1;
    PyObject* t2;

    sail::Tensor tensor1;
    sail::Tensor tensor2;

    if (!PyArg_ParseTuple(args, "OO", &t1, &t2)) {
        PyErr_SetString(PyExc_TypeError, "Inputs should be Sail Tensors");
        return NULL;
    }

    tensor1 = ((PyTensor*)t1)->tensor;
    tensor2 = ((PyTensor*)t2)->tensor;

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    sail::Tensor res = sail::ops::matmul(tensor1, tensor2);

    ret_class->tensor = res;
    ret_class->ndim = ((PyTensor*)t1)->ndim;
    ret_class->dtype = ((PyTensor*)t1)->dtype;

    return (PyObject*)ret_class;
}
