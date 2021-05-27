#pragma once
#include "../../src/modules/modules.h"

#include "py_module_def.h"

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
    // self->module = sail::modules::Linear(in_features, out_features,
    // use_bias);
    return 0;
}

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
    0,                                                 /* tp_call */
    0,                                                 /* tp_str */
    0,                                                 /* tp_getattro */
    0,                                                 /* tp_setattro */
    0,                                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,          /* tp_flags */
    "Custom objects",                /* tp_doc */
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
    (initproc)PyLinearModule_init,   /* tp_init */
    0,                               /* tp_alloc */
    PyModule_new,                    /* tp_new */
    PyObject_GC_Del                  /* tp_free */

};