// allow-no-source

#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../arg_parser.h"
#include "../py_dtypes/py_dtype.h"
#include "core/Tensor.h"
#include "core/TensorBody.h"
#include "core/dtypes.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/tensor_iterator.h"
#include "core/tensor_shape.h"
#include "core/types.h"
#include "numpy/arrayobject.h"

#include "../error_defs.h"
#include "../macros.h"

#define CAST_TYPE_CHECK(args, x)                              \
    {                                                         \
        if (!PyArg_ParseTuple(args, "O", &PyDtypeBase, &x)) { \
            return nullptr;                                   \
        }                                                     \
    }

static int PyTensor_init(PyTensor *self, PyObject *args, PyObject *kwargs) {
    START_EXCEPTION_HANDLING
    PyArrayObject *array;
    bool requires_grad = false;
    static char *kwlist[] = {"array", "requires_grad", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|b", kwlist, &array,
                                     &requires_grad)) {
        PyErr_SetString(PyExc_TypeError,
                        "must pass a tensor and a bool for requires_grad");
        return -1;
    }

    int ndim = PyArray_NDIM(array);
    int dtype = PyArray_TYPE(array);

    void *data = static_cast<void *>(array->data);
    Dtype dt = GetDtypeFromNumpyInt(dtype);
    int dt_size = GetDtypeSize(dt);
    TensorSize shape, strides;

    long int *shape_ptr = PyArray_SHAPE(array);

    for (int i = 0; i < ndim; i++) {
        shape.push_back(shape_ptr[i]);
    }

    self->tensor = sail::from_data(data, dt, sail::TensorShape(shape));
    self->tensor.requires_grad = requires_grad;

    return 0;
    END_EXCEPTION_HANDLING_INT
}
static int PyTensor_traverse(PyTensor *self, visitproc visit, void *arg) {
    Py_VISIT(self->base_object);
    return 0;
}

static int PyTensor_clear(PyTensor *self) {
    if (self->base_object != NULL) {
        Py_DECREF(self->base_object);
    }
    self->tensor.~Tensor();  // explicity call tensor destructor
    return 0;
}

static void PyTensor_dealloc(PyTensor *self) {
    PyObject_GC_UnTrack(self);
    PyTensor_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyTensor_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwds) {
    PyTensor *self;
    self = (PyTensor *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static PyObject *PyTensor_repr(PyTensor *self) {
    START_EXCEPTION_HANDLING
    return PyUnicode_FromString(sail::ops::tensor_repr(self->tensor).c_str());
    END_EXCEPTION_HANDLING
}

static PyObject *PyTensor_get_ndim(PyTensor *self, void *closure) {
    START_EXCEPTION_HANDLING
    long x = static_cast<long>(self->tensor.get_ndim());
    return PyLong_FromLong(x);
    END_EXCEPTION_HANDLING
}

static int PyTensor_set_ndim(PyTensor *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_AttributeError, "ndim cannot be set");
    return -1;
}

inline PyObject *inner_numpy(sail::Tensor &tensor) {
    START_EXCEPTION_HANDLING
    int ndims = tensor.get_ndim();
    long int *shape = tensor.get_shape_ptr();

    int type = tensor.get_np_type_num();

    PyObject *array;
    if (tensor.is_scalar()) {
        void *data = malloc(tensor.get_info().dtype_size);

        memcpy(data, tensor.get_data(), tensor.get_info().dtype_size);
        long shape = 0;
        array = PyArray_SimpleNewFromData(ndims, &shape, type, data);
    } else if (!tensor.is_view()) {
        void *data = malloc(tensor.getTotalSize());

        memcpy(data, tensor.get_data(), tensor.getTotalSize());
        array = PyArray_SimpleNewFromData(ndims, shape, type, data);
    } else {
        long numel = tensor.get_shape().numel();
        void *new_data = malloc(numel * tensor.get_info().dtype_size);
        dispatch_all_numeric_types(tensor.get_dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            T *data = (T *)tensor.get_data();
            T *data2 = (T *)new_data;
            sail::TensorShape s0 = tensor.get_shape();
            sail::MultiTensorIterator iter = sail::MultiTensorIterator(s0);
            long out_size = iter.out_loop_size();
            long in_size = iter.inner_loop_size();
            int z = 0;
            for (int i = 0; i < out_size; i++) {
                for (int j = 0; j < in_size; j++) {
                    data2[z] = data[iter.d_ptrs[0]];
                    iter.advance_d_ptr(1);
                    z += 1;
                }
                iter.backup_d_ptr();
                iter.next();
            }
        });
        shape = tensor.get_shape_ptr();
        ndims = tensor.get_ndim();
        array = PyArray_SimpleNewFromData(ndims, shape, type, new_data);
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)array, NPY_ARRAY_OWNDATA);
    return array;
    END_EXCEPTION_HANDLING
}
static PyObject *PyTensor_get_numpy(PyTensor *self, void *closure) {
    START_EXCEPTION_HANDLING

    PyObject *array = inner_numpy(self->tensor);

    return PyArray_Return((PyArrayObject *)array);
    END_EXCEPTION_HANDLING
}

static PyObject *PyTensor_get_grad(PyTensor *self, void *closure) {
    START_EXCEPTION_HANDLING
    if (self->tensor.has_grad() == false) {
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        PyTensor *grad;
        grad = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);
        SCTensor grad_ = self->tensor.get_grad();
        SCTensor gr = grad_;
        grad->tensor = gr;
        SET_BASE(self, grad);
        return (PyObject *)grad;
    }
    END_EXCEPTION_HANDLING
}

static int PyTensor_set_grad(PyTensor *self, void *closure) {
    PyErr_SetString(PyExc_AttributeError, "Grad cannot be modified");
    return -1;
}

static PyObject *PyTensor_get_requires_grad(PyTensor *self, void *closure) {
    START_EXCEPTION_HANDLING
    if (self->tensor.requires_grad) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
    END_EXCEPTION_HANDLING
}

static int PyTensor_set_requires_grad(PyTensor *self, PyObject *value,
                                      void *closure) {
    START_EXCEPTION_HANDLING
    if (PyObject_IsTrue(value)) {
        self->tensor.requires_grad = true;
    } else if (!PyObject_IsTrue(value)) {
        self->tensor.requires_grad = false;
    } else {
        PyErr_SetString(PyExc_AttributeError,
                        "requires_grad must be a boolean");
        return -1;
    }
    return 0;
    END_EXCEPTION_HANDLING_INT
}

static PyObject *PyTensor_astype(PyObject *self, PyObject *args,
                                 void *closure) {
    START_EXCEPTION_HANDLING
    PyObject *type;

    if (!PyArg_ParseTuple(args, "O", &type)) {
        return nullptr;
    }

    Dtype dt = ((PyDtype *)type)->dtype;

    PyTensor *ret_class;
    ret_class = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = ((PyTensor *)self)->tensor.cast(dt);

    return (PyObject *)ret_class;
    END_EXCEPTION_HANDLING
}

static PyObject *PyTensor_get_shape(PyTensor *self, void *closure) {
    START_EXCEPTION_HANDLING
    PyObject *tuple = PyTuple_New(self->tensor.get_ndim());
    int c = 0;
    for (long s : self->tensor.get_shape().shape) {
        PyTuple_SetItem(tuple, c, PyLong_FromLong(s));
        c += 1;
    }
    return tuple;
    END_EXCEPTION_HANDLING
}
static int PyTensor_set_shape(PyTensor *self, void *closure) {
    PyErr_SetString(PyExc_AttributeError,
                    "Shape cannot be modified like this. Use reshape");
    return -1;
}

static PyObject *PyTensor_backward(PyTensor *self, void *closure) {
    START_EXCEPTION_HANDLING
    self->tensor.backward();
    Py_INCREF(Py_None);
    return Py_None;
    END_EXCEPTION_HANDLING
}

static long PyTensor_len(PyTensor *self) { return self->tensor.len(); }

static PyObject *PyTensor_RichCompare(PyObject *self, PyObject *other, int op) {
    START_EXCEPTION_HANDLING
    sail::Tensor tensor1;
    sail::Tensor tensor2;

    if (!PyObject_TypeCheck(self, &PyTensorType) &&
        !PyObject_TypeCheck(other, &PyTensorType)) {
        return nullptr;
    }

    PyTensor *ret_class;
    ret_class = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    if (PyObject_TypeCheck(self, &PyTensorType) &&
        PyObject_TypeCheck(other, &PyTensorType)) {
        tensor1 = ((PyTensor *)self)->tensor;
        tensor2 = ((PyTensor *)other)->tensor;

        if (op == Py_LT) {
            ret_class->tensor = tensor1 < tensor2;
        } else if (op == Py_LE) {
            ret_class->tensor = tensor1 <= tensor2;
        } else if (op == Py_EQ) {
            ret_class->tensor = tensor1 == tensor2;
        } else if (op == Py_NE) {
            ret_class->tensor = tensor1 != tensor2;
        } else if (op == Py_GT) {
            ret_class->tensor = tensor1 > tensor2;
        } else if (op == Py_GE) {
            ret_class->tensor = tensor1 >= tensor2;
        }
    }

    if (PyObject_TypeCheck(self, &PyTensorType) &&
        PyObject_TypeCheck(other, &PyLong_Type)) {
        tensor1 = ((PyTensor *)self)->tensor;
        long val = PyLong_AsLong(other);

        if (op == Py_LT) {
            ret_class->tensor = tensor1 < val;
        } else if (op == Py_LE) {
            ret_class->tensor = tensor1 <= val;
        } else if (op == Py_EQ) {
            ret_class->tensor = tensor1 == val;
        } else if (op == Py_NE) {
            ret_class->tensor = tensor1 != val;
        } else if (op == Py_GT) {
            ret_class->tensor = tensor1 > val;
        } else if (op == Py_GE) {
            ret_class->tensor = tensor1 >= val;
        }
    }

    if (PyObject_TypeCheck(self, &PyLong_Type) &&
        PyObject_TypeCheck(other, &PyTensorType)) {
        long val = PyLong_AsLong(self);
        tensor2 = ((PyTensor *)other)->tensor;

        if (op == Py_LT) {
            ret_class->tensor = val < tensor2;
        } else if (op == Py_LE) {
            ret_class->tensor = val <= tensor2;
        } else if (op == Py_EQ) {
            ret_class->tensor = val == tensor2;
        } else if (op == Py_NE) {
            ret_class->tensor = val != tensor2;
        } else if (op == Py_GT) {
            ret_class->tensor = val > tensor2;
        } else if (op == Py_GE) {
            ret_class->tensor = val >= tensor2;
        }
    }

    if (PyObject_TypeCheck(self, &PyTensorType) &&
        PyObject_TypeCheck(other, &PyFloat_Type)) {
        tensor1 = ((PyTensor *)self)->tensor;
        double val = (double)PyFloat_AsDouble(other);

        if (op == Py_LT) {
            ret_class->tensor = tensor1 < val;
        } else if (op == Py_LE) {
            ret_class->tensor = tensor1 <= val;
        } else if (op == Py_EQ) {
            ret_class->tensor = tensor1 == val;
        } else if (op == Py_NE) {
            ret_class->tensor = tensor1 != val;
        } else if (op == Py_GT) {
            ret_class->tensor = tensor1 > val;
        } else if (op == Py_GE) {
            ret_class->tensor = tensor1 >= val;
        }
    }

    if (PyObject_TypeCheck(self, &PyFloat_Type) &&
        PyObject_TypeCheck(other, &PyTensorType)) {
        double val = (double)PyFloat_AsDouble(self);
        tensor2 = ((PyTensor *)other)->tensor;

        if (op == Py_LT) {
            ret_class->tensor = val < tensor2;
        } else if (op == Py_LE) {
            ret_class->tensor = val <= tensor2;
        } else if (op == Py_EQ) {
            ret_class->tensor = val == tensor2;
        } else if (op == Py_NE) {
            ret_class->tensor = val != tensor2;
        } else if (op == Py_GT) {
            ret_class->tensor = val > tensor2;
        } else if (op == Py_GE) {
            ret_class->tensor = val >= tensor2;
        }
    }

    return (PyObject *)ret_class;
    END_EXCEPTION_HANDLING
}