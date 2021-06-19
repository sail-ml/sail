#pragma once
#include "../error_defs.h"
#include "../macros.h"
#include "../py_tensor/py_tensor_def.h"
#include "core/loss/mean_squared_error.h"
#include "core/modules/modules.h"

#include "py_loss_def.h"

using Module = sail::modules::Module;

static int PyMSELoss_init(PyLoss *self, PyObject *args, PyObject *kwargs) {
    self->module = (Module *)(new sail::loss::MeanSquaredError());
    return 0;
}

RETURN_OBJECT
PyMSELoss_forward(PyLoss *self, PyObject *args, PyObject *kwargs) {
    START_EXCEPTION_HANDLING
    PyTensor *logits = NULL;
    PyTensor *targets = NULL;
    static char *kwlist[] = {"logits", "targets", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &logits,
                                     &targets)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return nullptr;
    }
    PyTensor *py_output = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    sail::Tensor output = ((sail::loss::MeanSquaredError *)(self->module))
                              ->forward(logits->tensor, targets->tensor);
    GENERATE_FROM_TENSOR(py_output, output);
    return (PyObject *)py_output;
    END_EXCEPTION_HANDLING
}

static PyMethodDef PyMSELoss_methods[] = {
    {"forward", (PyCFunction)PyMSELoss_forward, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject PyMSELossType = {
    PyVarObject_HEAD_INIT(NULL, 0) "libsail_c.MeanSquaredError", /* tp_name */
    sizeof(PyLoss),             /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)PyLoss_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_reserved */
    0,                          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    PyMSELoss_forward,          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,        /* tp_flags */
    NULL,                          /* tp_doc */
    (traverseproc)PyLoss_traverse, /* tp_traverse */
    (inquiry)PyLoss_clear,         /* tp_clear */
    0,                             /* tp_richcompare */
    0,                             /* tp_weaklistoffset */
    0,                             /* tp_iter */
    0,                             /* tp_iternext */
    PyMSELoss_methods,             /* tp_methods */
    0,                             /* tp_members */
    0,                             // PyLoss_getsetters, /* tp_getset */
    &PyLossType,                   /* tp_base */
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    0,                             /* tp_dictoffset */
    (initproc)PyMSELoss_init,      /* tp_init */
    0,                             /* tp_alloc */
    PyLoss_new,                    /* tp_new */
    PyObject_GC_Del                /* tp_free */

};