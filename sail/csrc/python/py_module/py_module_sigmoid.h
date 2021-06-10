#pragma once
#include "../../core/modules/modules.h"
#include "../macros.h"
#include "../py_tensor/py_tensor_def.h"

#include "py_module_def.h"

using Sigmoid = sail::modules::Sigmoid;

static int PySigmoidModule_init(PyModule *self, PyObject *args,
                                PyObject *kwargs) {
    self->module = (Module *)(new Sigmoid());
    return 0;
}

RETURN_OBJECT
PySigmoidModule_forward(PyModule *self, PyObject *args, PyObject *kwargs) {
    PyTensor *inputs = NULL;
    static char *kwlist[] = {"inputs", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &inputs)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }
    PyTensor *py_output = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    sail::Tensor output = ((Sigmoid *)(self->module))->forward(inputs->tensor);
    GENERATE_FROM_TENSOR(py_output, output);
    return (PyObject *)py_output;
}

static PyMethodDef PySigmoidModule_methods[] = {
    {"forward", (PyCFunction)PySigmoidModule_forward,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject PySigmoidModuleType = {
    PyVarObject_HEAD_INIT(NULL, 0) "libsail_c.Linear", /* tp_name */
    sizeof(PyModule),                                  /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)PyModule_dealloc,                      /* tp_dealloc */
    0,                                                 /* tp_print */
    0,                                                 /* tp_getattr */
    0,                                                 /* tp_setattr */
    0,                                                 /* tp_reserved */
    0,                                                 /* tp_repr */
    0,                                                 /* tp_as_number */
    0,                                                 /* tp_as_sequence */
    0,                                                 /* tp_as_mapping */
    0,                                                 /* tp_hash */
    PySigmoidModule_forward,                           /* tp_call */
    0,                                                 /* tp_str */
    0,                                                 /* tp_getattro */
    0,                                                 /* tp_setattro */
    0,                                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,          /* tp_flags */
    NULL,                            /* tp_doc */
    (traverseproc)PyModule_traverse, /* tp_traverse */
    (inquiry)PyModule_clear,         /* tp_clear */
    0,                               /* tp_richcompare */
    0,                               /* tp_weaklistoffset */
    0,                               /* tp_iter */
    0,                               /* tp_iternext */
    0,                               /* tp_methods */
    0,                               /* tp_members */
    0,                               // PyModule_getsetters, /* tp_getset */
    &PyModuleType,                   /* tp_base */
    0,                               /* tp_dict */
    0,                               /* tp_descr_get */
    0,                               /* tp_descr_set */
    0,                               /* tp_dictoffset */
    (initproc)PySigmoidModule_init,  /* tp_init */
    0,                               /* tp_alloc */
    PyModule_new,                    /* tp_new */
    PyObject_GC_Del                  /* tp_free */

};