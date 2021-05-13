#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../../src/Tensor.h"
#include "../../src/TensorBody.h"
#include "../../src/dtypes.h"
#include "../../src/factories.h"
#include "../../src/ops/ops.h"
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

    void *data = static_cast<void *>(array->data);
    Dtype dt = GetDtypeFromNumpyInt(dtype);
    int dt_size = GetDtypeSize(dt);
    TensorSize shape, strides;

    long int *shape_ptr = PyArray_SHAPE(array);
    long int *stride_ptr = PyArray_STRIDES(array);

    for (int i = 0; i < ndim; i++) {
        shape.push_back(shape_ptr[i]);
        strides.push_back(stride_ptr[i] / dt_size);
    }

    // SCTensor tensor = SCTensor(
    //     ndim, data, dt, sail::TensorShape(shape, strides), requires_grad);
    // sail::TensorBody *b =
    //     new sail::TensorBody(data, dt, sail::TensorShape(shape, strides));
    // self->tensor = SCTensor(b, requires_grad);
    self->tensor = sail::from_data(data, dt, sail::TensorShape(shape));
    self->ndim = ndim;
    self->requires_grad = requires_grad;
    self->dtype = dtype;

    return 0;
}
static int PyTensor_traverse(PyTensor *self, visitproc visit, void *arg) {
    // Py_VISIT(self->base_object);
    return 0;
}

static int PyTensor_clear(PyTensor *self) {
    // std::cout << "PY FREE" << std::endl;
    self->tensor.~Tensor();  // explicity call tensor destructor
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

RETURN_OBJECT PyTensor_repr(PyTensor *self) {
    return PyUnicode_FromString(sail::ops::tensor_repr(self->tensor).c_str());
}

RETURN_OBJECT PyTensor_get_ndim(PyTensor *self, void *closure) {
    long x = static_cast<long>(self->ndim);
    return PyLong_FromLong(x);
}
PyObject *inner_numpy(sail::Tensor &tensor) {
    int ndims = tensor.get_ndim();
    long int *shape = tensor.get_shape_ptr();

    int type = tensor.get_np_type_num();
    void *data = malloc(tensor.getTotalSize());  // self->tensor.data;

    memcpy(data, tensor.get_data(), tensor.getTotalSize());
    PyObject *array;
    if (!tensor.is_view()) {
        array = PyArray_SimpleNewFromData(ndims, shape, type, data);
    } else {
        long numel = tensor.get_shape().numel();
        void *new_data = malloc(numel * tensor.get_info().dtype_size);
        launch_arithmetic(tensor.get_dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            T *data = (T *)tensor.get_data();
            T *data2 = (T *)new_data;
            sail::TensorShape s0 = tensor.get_shape();
            for (int i = 0; i < numel; i++) {
                data2[i] = data[s0.d_ptr];
                s0.next();
            }
            s0.reset();
        });
        shape = tensor.get_shape().get_shape_ptr();
        ndims = tensor.get_shape().ndim();
        array = PyArray_SimpleNewFromData(ndims, shape, type, new_data);
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)array, NPY_ARRAY_OWNDATA);
    return array;
}
RETURN_OBJECT
PyTensor_get_numpy(PyTensor *self, void *closure) {
    // Py_INCREF(self);
    PyObject *array = inner_numpy(self->tensor);

    // PyArray_SetBaseObject((PyArrayObject *)array, (PyObject *)self);
    return PyArray_Return((PyArrayObject *)array);
}

RETURN_OBJECT PyTensor_get_grad(PyTensor *self, void *closure) {
    if (self->tensor.has_grad == false) {
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        // Py_INCREF(self);
        PyTensor *grad;
        grad = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

        SCTensor gr = clone(*(self->tensor.grad));
        // self->tensor.grad->owner = false;
        grad->tensor = std::move(gr);
        grad->ndim = grad->tensor.get_ndim();
        grad->dtype = self->dtype;
        // SET_BASE(self, grad);
        return (PyObject *)grad;
    }
}

static int PyTensor_set_grad(PyTensor *self, void *closure) {
    PyErr_SetString(PyExc_AttributeError, "Grad cannot be modified");
    return -1;
}

RETURN_OBJECT
PyTensor_astype(PyObject *self, PyObject *args, void *closure) {
    PyDtype *type;

    CAST_TYPE_CHECK(args, type);

    Dtype dt = ((PyDtype *)type)->dtype;

    PyTensor *ret_class;
    ret_class = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    ret_class->tensor = ((PyTensor *)self)->tensor.cast(dt);

    ret_class->ndim = ret_class->tensor.get_ndim();
    ret_class->dtype = ((PyDtype *)type)->dt_val;

    return (PyObject *)ret_class;
}

RETURN_OBJECT
PyTensor_get_shape(PyTensor *self, void *closure) {
    PyObject *tuple = PyTuple_New(self->tensor.get_ndim());
    int c = 0;
    for (long s : self->tensor.get_shape().shape) {
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
    Py_INCREF(Py_None);
    return Py_None;
}
