#pragma once
#include "../macros.h"
#include "../py_tensor/py_tensor_def.h"
#include "core/modules/modules.h"

#include "py_module_def.h"

using Module = sail::modules::Module;
using Linear = sail::modules::Linear;

static int PyLinearModule_init(PyModule *self, PyObject *args,
                               PyObject *kwargs) {
    int in_features, out_features;
    bool use_bias = true;
    static char *kwlist[] = {"in_features", "out_features", "use_bias", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|b", kwlist, &in_features,
                                     &out_features, &use_bias)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }

    // self->module = sail::modules::Module();
    self->module = (Module *)(new sail::modules::Linear(
        in_features, out_features, use_bias));
    return 0;
}

RETURN_OBJECT
PyLinearModule_get_weights(PyModule *self, void *closure) {
    PyTensor *py_weights = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);
    // Linear a = *(Linear *)self->module;
    Tensor weights = (*(Linear *)self->module).weights;
    GENERATE_FROM_TENSOR(py_weights, weights);

    return (PyObject *)py_weights;
}

static int PyLinearModule_set_weights(PyModule *self, PyTensor *t,
                                      void *closure) {
    ((Linear *)(self->module))->weights = t->tensor;
    return 0;
}

RETURN_OBJECT
PyLinearModule_get_bias(PyModule *self, void *closure) {
    PyTensor *py_bias = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);
    Linear a = *(Linear *)self->module;
    GENERATE_FROM_TENSOR(py_bias, a.biases);
    return (PyObject *)py_bias;
}

static int PyLinearModule_set_bias(PyModule *self, PyTensor *t, void *closure) {
    ((Linear *)(self->module))->biases = t->tensor;
    return 0;
}

static PyGetSetDef PyLinearModule_get_setters[] = {
    {"weights", (getter)PyLinearModule_get_weights,
     (setter)PyLinearModule_set_weights, NULL},
    {"bias", (getter)PyLinearModule_get_bias, (setter)PyLinearModule_set_bias,
     NULL},
    {NULL} /* Sentinel */
};

RETURN_OBJECT
PyLinearModule_forward(PyModule *self, PyObject *args, PyObject *kwargs) {
    PyTensor *inputs = NULL;
    static char *kwlist[] = {"inputs", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &inputs)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
    }
    PyTensor *py_output = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    sail::Tensor output = ((Linear *)(self->module))->forward(inputs->tensor);
    GENERATE_FROM_TENSOR(py_output, output);
    return (PyObject *)py_output;
}

static PyMethodDef PyLinearModule_methods[] = {
    {"forward", (PyCFunction)PyLinearModule_forward,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject PyLinearModuleType = {
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
    PyLinearModule_forward,                            /* tp_call */
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
    PyLinearModule_methods,          /* tp_methods */
    0,                               /* tp_members */
    PyLinearModule_get_setters,      // PyModule_getsetters, /* tp_getset */
    &PyModuleType,                   /* tp_base */
    0,                               /* tp_dict */
    0,                               /* tp_descr_get */
    0,                               /* tp_descr_set */
    0,                               /* tp_dictoffset */
    (initproc)PyLinearModule_init,   /* tp_init */
    0,                               /* tp_alloc */
    PyModule_new,                    /* tp_new */
    PyObject_GC_Del                  /* tp_free */

};