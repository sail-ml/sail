#pragma once

#include <Python.h>
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "py_tensor/py_tensor.h"

// sail::Tensor getNumeric(PyObject* number) {
//     sail::Tensor new_tensor;
//     if (PyObject_TypeCheck(number, &PyFloat_Type)) {

//         new_tensor = sail::empty_scalar(Dtype::sFloat64);
//         double val = PyFloat_AsDouble(number);
//         double *ptr = &val;
//         memcpy(new_tensor.storage.data, ptr, sizeof(val));
//         return new_tensor;
//     } else if (PyObject_TypeCheck(number, &PyLong_Type)) {
//         new_tensor = sail::empty_scalar(Dtype::sFloat64);
//         new_tensor.free();
//         new_tensor.storage.data = PyLong_AsVoidPtr(number);
//     }

//     return new_tensor;
// }
// new_tensor = sail::empty_scalar(Dtype::sFloat64);  \
            // new_tensor.free();                                 \
            // new_tensor.set_data(PyLong_AsVoidPtr(number));     \

#define GET_NUMERIC(number, new_tensor)                        \
    {                                                          \
        if (PyObject_TypeCheck(number, &PyFloat_Type)) {       \
            new_tensor = sail::empty_scalar(Dtype::sFloat64);  \
            double val = PyFloat_AsDouble(number);             \
            double *ptr = &val;                                \
            memcpy(new_tensor.get_data(), ptr, sizeof(val));   \
        } else if (PyObject_TypeCheck(number, &PyLong_Type)) { \
            PyErr_SetString(PyExc_TypeError, "Nah.");          \
        }                                                      \
    }

#define RETURN_OBJECT static PyObject *
// using RETURN_OBJECT = RETURN_OBJECT;

#define BINARY_TENSOR_TYPE_CHECK(a, b)               \
    {                                                \
        if (!PyObject_TypeCheck(a, &PyTensorType)) { \
            return NULL;                             \
        }                                            \
        if (!PyObject_TypeCheck(b, &PyTensorType)) { \
            return NULL;                             \
        }                                            \
    }

#define COPY(src, dest)                      \
    {                                        \
        dest->tensor = src->tensor;          \
        dest->ndim = src->ndim;              \
        dest->dtype = src->dtype;            \
        dest->base_object = (PyObject *)src; \
        Py_INCREF(src);                      \
    }
#define SET_BASE(src, dest)                  \
    {                                        \
        Py_INCREF(src);                      \
        dest->base_object = (PyObject *)src; \
    }

#define NUMERIC_PROCESS(t1, t2)                              \
    {                                                        \
        if (!PyObject_TypeCheck(t1, &PyTensorType) &&        \
            !PyObject_TypeCheck(t2, &PyTensorType)) {        \
            return NULL;                                     \
        }                                                    \
        if (PyObject_TypeCheck(t1, &PyTensorType) &&         \
            PyObject_TypeCheck(t2, &PyTensorType)) {         \
            tensor1 = ((PyTensor *)t1)->tensor;              \
            tensor2 = ((PyTensor *)t2)->tensor;              \
        } else if (PyObject_TypeCheck(t1, &PyTensorType) &&  \
                   !PyObject_TypeCheck(t2, &PyTensorType)) { \
            tensor1 = ((PyTensor *)t1)->tensor;              \
            GET_NUMERIC(t2, tensor2);                        \
        } else if (!PyObject_TypeCheck(t1, &PyTensorType) && \
                   PyObject_TypeCheck(t2, &PyTensorType)) {  \
            tensor1 = ((PyTensor *)t2)->tensor;              \
            GET_NUMERIC(t1, tensor2);                        \
        } else {                                             \
            return NULL;                                     \
        }                                                    \
    }

#define GENERATE_FROM_TENSOR(pyobj, t)                          \
    {                                                           \
        pyobj->tensor = t;                                      \
        pyobj->ndim = t.get_ndim();                             \
        pyobj->dtype = get_np_type_numFromDtype(t.get_dtype()); \
        pyobj->requires_grad = t.requires_grad;                 \
    }