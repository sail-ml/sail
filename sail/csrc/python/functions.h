
#pragma once

#include <Python.h>
#include <structmember.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "py_tensor/py_tensor.h"
#include "core/Tensor.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "numpy/arrayobject.h"

#define REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims)           \
    {                                                                       \
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist, &t1, \
                                         &axis, &keepdims)) {               \
            PyErr_SetString(PyExc_TypeError,                                \
                            "must pass a tensor and an integer for axis");  \
            return nullptr;                                                 \
        }                                                                   \
    }



PyObject* sail_add(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    PyTensor * x2;
    static char* kwlist[] = { "x1", "x2", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &x1, &x2)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::add(x1->tensor, x2->tensor);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_subtract(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    PyTensor * x2;
    static char* kwlist[] = { "x1", "x2", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &x1, &x2)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::subtract(x1->tensor, x2->tensor);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_divide(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    PyTensor * x2;
    static char* kwlist[] = { "x1", "x2", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &x1, &x2)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::divide(x1->tensor, x2->tensor);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_multiply(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    PyTensor * x2;
    static char* kwlist[] = { "x1", "x2", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &x1, &x2)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::multiply(x1->tensor, x2->tensor);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_power(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    PyTensor * x2;
    static char* kwlist[] = { "x1", "x2", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &x1, &x2)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::power(x1->tensor, x2->tensor);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_matmul(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    PyTensor * x2;
    static char* kwlist[] = { "x1", "x2", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &x1, &x2)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::matmul(x1->tensor, x2->tensor);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_addmm(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    PyTensor * x2;
    PyTensor * x3;
    static char* kwlist[] = { "x1", "x2", "x3", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &x1, &x2, &x3)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::addmm(x1->tensor, x2->tensor, x3->tensor);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_log(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    static char* kwlist[] = { "x1", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &x1)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::log(x1->tensor);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_exp(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    static char* kwlist[] = { "x1", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &x1)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::exp(x1->tensor);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_clip(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * x1;
    double min;
    double max;
    static char* kwlist[] = { "x1", "min", "max", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "Odd", kwlist, &x1, &min, &max)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = sail::ops::clip(x1->tensor, min, max);
        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_sum(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor* t1;
    PyObject* axis = Py_None;
    int keepdims = 0;
    static char* kwlist[] = {"tensor", "axis", "keepdims", NULL};

    REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims);

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    if (axis == Py_None) {
        ret_class->tensor =
            sail::ops::sum(((PyTensor*)t1)->tensor, NULLDIM, (bool)keepdims);
    } else {
        ret_class->tensor = sail::ops::sum(((PyTensor*)t1)->tensor,
                                           PyLong_AsLong(axis), (bool)keepdims);
    }

    return (PyObject*)ret_class;

    END_EXCEPTION_HANDLING
}


PyObject* sail_mean(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor* t1;
    PyObject* axis = Py_None;
    int keepdims = 0;
    static char* kwlist[] = {"tensor", "axis", "keepdims", NULL};

    REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims);

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    if (axis == Py_None) {
        ret_class->tensor =
            sail::ops::mean(((PyTensor*)t1)->tensor, NULLDIM, (bool)keepdims);
    } else {
        ret_class->tensor = sail::ops::mean(((PyTensor*)t1)->tensor,
                                           PyLong_AsLong(axis), (bool)keepdims);
    }

    return (PyObject*)ret_class;

    END_EXCEPTION_HANDLING
}


PyObject* sail_max(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor* t1;
    PyObject* axis = Py_None;
    int keepdims = 0;
    static char* kwlist[] = {"tensor", "axis", "keepdims", NULL};

    REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims);

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    if (axis == Py_None) {
        ret_class->tensor =
            sail::ops::max(((PyTensor*)t1)->tensor, NULLDIM, (bool)keepdims);
    } else {
        ret_class->tensor = sail::ops::max(((PyTensor*)t1)->tensor,
                                           PyLong_AsLong(axis), (bool)keepdims);
    }

    return (PyObject*)ret_class;

    END_EXCEPTION_HANDLING
}


PyObject* sail_min(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor* t1;
    PyObject* axis = Py_None;
    int keepdims = 0;
    static char* kwlist[] = {"tensor", "axis", "keepdims", NULL};

    REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims);

    PyTensor* ret_class;
    ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

    if (axis == Py_None) {
        ret_class->tensor =
            sail::ops::min(((PyTensor*)t1)->tensor, NULLDIM, (bool)keepdims);
    } else {
        ret_class->tensor = sail::ops::min(((PyTensor*)t1)->tensor,
                                           PyLong_AsLong(axis), (bool)keepdims);
    }

    return (PyObject*)ret_class;

    END_EXCEPTION_HANDLING
}

static PyObject* sail_tensordot(PyObject* self, PyObject* args, PyObject* kwargs) {
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

static PyObject* sail_broadcast_to(PyObject* self, PyObject* args) {
  START_EXCEPTION_HANDLING
  PyTensor* t1;
  PyObject* tuple;

  if (!PyArg_ParseTuple(args, "OO", &t1, &tuple)) {
      PyErr_SetString(PyExc_TypeError, "Inputs should be Sail Tensors");
      return nullptr;
  }

  int len = PyTuple_Size(tuple);
  if (len == -1) {
      PyErr_SetString(PyExc_TypeError, "Shape must have atleat 1 element.");
      return nullptr;
  }
  std::vector<long> shape;
  while (len--) {
      shape.push_back(PyLong_AsLong(PyTuple_GetItem(tuple, len)));
  }
  std::reverse(shape.begin(), shape.end());

  sail::TensorShape s = sail::TensorShape(shape);

  PyTensor* ret_class;
  ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

  ret_class->tensor = sail::ops::broadcast_to(t1->tensor, s);

  // ret_class->dtype = t1- >ndim;
  SET_BASE(t1, ret_class);
  return (PyObject*)ret_class;
  END_EXCEPTION_HANDLING
}


PyObject* sail_reshape(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * t1;
    PyObject * new_shape;
    static char* kwlist[] = { "t1", "new_shape", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &t1, &new_shape)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        new_shape = PySequence_Tuple(new_shape);
int len = PyTuple_Size(new_shape);
if (len == -1) {
    PyErr_SetString(PyExc_TypeError, "Shape must have atleat 1 element.");
    return nullptr;
}
TensorSize size;
while (len--) {
    size.push_back(PyLong_AsLong(PyTuple_GetItem(new_shape, len)));
}

std::reverse(size.begin(), size.end());

sail::TensorShape new_ = sail::TensorShape(size);

ret_class->tensor = t1->tensor.reshape(new_);
ret_class->base_object = (PyObject*)t1;
Py_INCREF(t1);


        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_transpose(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * t1;
    PyObject * tuple = NULL;
    static char* kwlist[] = { "t1", "tuple", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &t1, &tuple)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        if (tuple == NULL) {
    ret_class->tensor = sail::ops::transpose(t1->tensor);
} else {
    tuple = PySequence_Tuple(tuple);
    int len = PyTuple_Size(tuple);
    if (len == -1) {
        PyErr_SetString(PyExc_TypeError,
                        "Shape must have atleat 1 element.");
        return nullptr;
    }
    std::vector<long> shape;
    while (len--) {
        shape.push_back(PyLong_AsLong(PyTuple_GetItem(tuple, len)));
    }
    std::reverse(shape.begin(), shape.end());

    ret_class->tensor = sail::ops::transpose(t1->tensor, shape);
}

ret_class->base_object = (PyObject*)t1;
Py_INCREF(ret_class->base_object);


        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_expand_dims(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * t1;
    int dim;
    static char* kwlist[] = { "t1", "dim", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &t1, &dim)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = t1->tensor.expand_dims(dim);
ret_class->base_object = (PyObject*)t1;
Py_INCREF(t1);

return (PyObject*)ret_class;


        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_squeeze(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * t1;
    int dim;
    static char* kwlist[] = { "t1", "dim", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &t1, &dim)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor = t1->tensor.squeeze(dim);
ret_class->base_object = (PyObject*)t1;
Py_INCREF(t1);

return (PyObject*)ret_class;


        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_rollaxis(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * t1;
    int axis = 0;
    int position = 0;
    static char* kwlist[] = { "t1", "axis", "position", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", kwlist, &t1, &axis, &position)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor =
          sail::ops::rollaxis(t1->tensor, axis, position);

  ret_class->base_object = (PyObject*)t1;
  Py_INCREF(ret_class->base_object);


        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_moveaxis(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyTensor * t1;
    int axis = 0;
    int position = 0;
    static char* kwlist[] = { "t1", "axis", "position", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", kwlist, &t1, &axis, &position)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        ret_class->tensor =
        sail::ops::moveaxis(t1->tensor, axis, position);

ret_class->base_object = (PyObject*)t1;
Py_INCREF(ret_class->base_object);


        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_cat(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyObject * tensors;
    int axis = 1;
    static char* kwlist[] = { "tensors", "axis", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &tensors, &axis)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        int len = PySequence_Length(tensors);
std::vector<sail::Tensor> tensor_list;
if (len < 2) {
  PyErr_SetString(PyExc_TypeError,
        "Tensor list must have atleast 2 elements.");
        return nullptr;
}
while(len--) {
  PyObject* x = PySequence_GetItem(tensors, len);
  tensor_list.push_back(((PyTensor*)x)->tensor);
}
std::reverse(tensor_list.begin(), tensor_list.end());

ret_class->tensor = sail::ops::cat(tensor_list, axis);


        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}


PyObject* sail_stack(PyObject* self, PyObject* args, PyObject* kwargs) {
    START_EXCEPTION_HANDLING
    
    PyObject * tensors;
    int axis = 1;
    static char* kwlist[] = { "tensors", "axis", NULL };
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &tensors, &axis)) {
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        int len = PySequence_Length(tensors);
std::vector<sail::Tensor> tensor_list;
if (len < 2) {
  PyErr_SetString(PyExc_TypeError,
        "Tensor list must have atleast 2 elements.");
        return nullptr;
}
while(len--) {
  PyObject* x = PySequence_GetItem(tensors, len);
  tensor_list.push_back(((PyTensor*)x)->tensor);
}
std::reverse(tensor_list.begin(), tensor_list.end());

ret_class->tensor = sail::ops::stack(tensor_list, axis);


        return (PyObject*)ret_class;
    }

    else {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }

    END_EXCEPTION_HANDLING
}

static PyObject*
sail_add_docstring(PyObject *unused, PyObject *args) {
    PyObject *obj;
    PyObject *str;
#if PY_VERSION_HEX >= 0x030700A2 && \
    (!defined(PYPY_VERSION_NUM) || PYPY_VERSION_NUM > 0x07030300)
    const char *docstr;
#else
    char *docstr;
#endif
    static char *msg = "already has a different docstring";

    /* Don't add docstrings */
    if (Py_OptimizeFlag > 1) {
        Py_RETURN_NONE;
    }

    if (!PyArg_ParseTuple(args, "OO!:add_docstring", &obj, &PyUnicode_Type,
                          &str)) {
        return nullptr;
    }

    docstr = PyUnicode_AsUTF8(str);
    if (docstr == NULL) {
        return nullptr;
    }

#define _ADDDOC(doc, name)                                           \
    if (!(doc)) {                                                    \
        doc = docstr;                                                \
        Py_INCREF(str); /* hold on to string (leaks reference) */    \
    } else if (strcmp(doc, docstr) != 0) {                           \
        PyErr_Format(PyExc_RuntimeError, "%s method %s", name, msg); \
        return nullptr;                                              \
    }

    if (Py_TYPE(obj) == &PyCFunction_Type) {
        PyCFunctionObject *new_ob = (PyCFunctionObject *)obj;
        _ADDDOC(new_ob->m_ml->ml_doc, new_ob->m_ml->ml_name);
    } else if (Py_TYPE(obj) == &PyType_Type) {
        PyTypeObject *new_ob = (PyTypeObject *)obj;
        _ADDDOC(new_ob->tp_doc, new_ob->tp_name);
    } else if (Py_TYPE(obj) == &PyMemberDescr_Type) {
        PyMemberDescrObject *new_ob = (PyMemberDescrObject *)obj;
        _ADDDOC(new_ob->d_member->doc, new_ob->d_member->name);
    } else if (Py_TYPE(obj) == &PyGetSetDescr_Type) {
        PyGetSetDescrObject *new_ob = (PyGetSetDescrObject *)obj;
        _ADDDOC(new_ob->d_getset->doc, new_ob->d_getset->name);
    } else if (Py_TYPE(obj) == &PyMethodDescr_Type) {
        PyMethodDescrObject *new_ob = (PyMethodDescrObject *)obj;
        _ADDDOC(new_ob->d_method->ml_doc, new_ob->d_method->ml_name);
    } else {
        PyObject *doc_attr;

        doc_attr = PyObject_GetAttrString(obj, "__doc__");
        if (doc_attr != NULL && doc_attr != Py_None &&
            (PyUnicode_Compare(doc_attr, str) != 0)) {
            Py_DECREF(doc_attr);
            if (PyErr_Occurred()) {
                /* error during PyUnicode_Compare */
                return nullptr;
            }
            PyErr_Format(PyExc_RuntimeError, "object %s", msg);
            return nullptr;
        }
        Py_XDECREF(doc_attr);

        if (PyObject_SetAttrString(obj, "__doc__", str) < 0) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot set a docstring for that object");
            return nullptr;
        }
        Py_RETURN_NONE;
    }

#undef _ADDDOC

    Py_RETURN_NONE;
}
static PyMethodDef SailOpsMethods[] = {
    {"add", (PyCFunction)sail_add, METH_VARARGS | METH_KEYWORDS, NULL},
    {"subtract", (PyCFunction)sail_subtract, METH_VARARGS | METH_KEYWORDS, NULL},
    {"divide", (PyCFunction)sail_divide, METH_VARARGS | METH_KEYWORDS, NULL},
    {"multiply", (PyCFunction)sail_multiply, METH_VARARGS | METH_KEYWORDS, NULL},
    {"power", (PyCFunction)sail_power, METH_VARARGS | METH_KEYWORDS, NULL},
    {"matmul", (PyCFunction)sail_matmul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"addmm", (PyCFunction)sail_addmm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"log", (PyCFunction)sail_log, METH_VARARGS | METH_KEYWORDS, NULL},
    {"exp", (PyCFunction)sail_exp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clip", (PyCFunction)sail_clip, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sum", (PyCFunction)sail_sum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mean", (PyCFunction)sail_mean, METH_VARARGS | METH_KEYWORDS, NULL},
    {"max", (PyCFunction)sail_max, METH_VARARGS | METH_KEYWORDS, NULL},
    {"min", (PyCFunction)sail_min, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tensordot", (PyCFunction)sail_tensordot, METH_VARARGS | METH_KEYWORDS, NULL},
    {"broadcast_to", (PyCFunction)sail_broadcast_to, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reshape", (PyCFunction)sail_reshape, METH_VARARGS | METH_KEYWORDS, NULL},
    {"transpose", (PyCFunction)sail_transpose, METH_VARARGS | METH_KEYWORDS, NULL},
    {"expand_dims", (PyCFunction)sail_expand_dims, METH_VARARGS | METH_KEYWORDS, NULL},
    {"squeeze", (PyCFunction)sail_squeeze, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rollaxis", (PyCFunction)sail_rollaxis, METH_VARARGS | METH_KEYWORDS, NULL},
    {"moveaxis", (PyCFunction)sail_moveaxis, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cat", (PyCFunction)sail_cat, METH_VARARGS | METH_KEYWORDS, NULL},
    {"stack", (PyCFunction)sail_stack, METH_VARARGS | METH_KEYWORDS, NULL},
    {"add_docstring", (PyCFunction)sail_add_docstring, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL} };
