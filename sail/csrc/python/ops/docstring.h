#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../py_tensor/py_tensor.h"
#include "core/Tensor.h"
#include "core/ops/reduction.h"
#include "numpy/arrayobject.h"

#include "../macros.h"

RETURN_OBJECT
add_docstring(PyObject *unused, PyObject *args) {
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