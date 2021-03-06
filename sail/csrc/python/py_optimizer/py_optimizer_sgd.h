#pragma once
#include "../error_defs.h"
#include "../macros.h"
#include "../py_tensor/py_tensor_def.h"
#include "core/loss/cross_entropy_loss.h"
#include "core/modules/modules.h"
#include "core/optimizers/optimizers.h"

#include "py_optimizer_def.h"

using SGD = sail::optimizers::SGD;
using Optimizer = sail::optimizers::Optimizer;

static int PyOptimizerSGD_init(PyOptimizer *self, PyObject *args,
                               PyObject *kwargs) {
    START_EXCEPTION_HANDLING
    // self->module = sail::modules::Module();
    float learning_rate = 0.0001;
    static char *kwlist[] = {"learning_rate", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f", kwlist,
                                     &learning_rate)) {
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return -1;
    }
    self->optimizer = (Optimizer *)(new SGD(learning_rate));
    return 0;
    END_EXCEPTION_HANDLING_INT
}

// static PyObject*
// PyOptimizerSGD_forward(PyOptimizer *self, PyObject *args, PyObject *kwargs) {
//     PyTensor *logits = NULL;
//     PyTensor *targets = NULL;
//     static char *kwlist[] = {"logits", "targets", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &logits,
//                                      &targets)) {
//         PyErr_SetString(PyExc_TypeError, "incorrect arguments");
//     }
//     PyTensor *py_output = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType,
//     0);

//     sail::Tensor output =
//         ((Loss *)(self->module))->forward(logits->tensor, targets->tensor);
//     GENERATE_FROM_TENSOR(py_output, output);
//     return (PyObject *)py_output;
// }

// static PyMethodDef PyOptimizer_methods[] = {
//     {"forward", (PyCFunction)PyOptimizer_forward, METH_VARARGS |
//     METH_KEYWORDS,
//      NULL},
//     {NULL} /* Sentinel */
// };

static PyTypeObject PyOptimizerSGDType = {
    PyVarObject_HEAD_INIT(NULL, 0) "sail.optimizers.SGD", /* tp_name */
    sizeof(PyOptimizer),                                  /* tp_basicsize */
    0,                                                    /* tp_itemsize */
    (destructor)PyOptimizer_dealloc,                      /* tp_dealloc */
    0,                                                    /* tp_print */
    0,                                                    /* tp_getattr */
    0,                                                    /* tp_setattr */
    0,                                                    /* tp_reserved */
    0,                                                    /* tp_repr */
    0,                                                    /* tp_as_number */
    0,                                                    /* tp_as_sequence */
    0,                                                    /* tp_as_mapping */
    0,                                                    /* tp_hash */
    0,                                                    /* tp_call */
    0,                                                    /* tp_str */
    0,                                                    /* tp_getattro */
    0,                                                    /* tp_setattro */
    0,                                                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,             /* tp_flags */
    NULL,                               /* tp_doc */
    (traverseproc)PyOptimizer_traverse, /* tp_traverse */
    (inquiry)PyOptimizer_clear,         /* tp_clear */
    0,                                  /* tp_richcompare */
    0,                                  /* tp_weaklistoffset */
    0,                                  /* tp_iter */
    0,                                  /* tp_iternext */
    0,                                  /* tp_methods */
    0,                                  /* tp_members */
    0,                             // PyOptimizer_getsetters, /* tp_getset */
    &PyOptimizerType,              /* tp_base */
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    0,                             /* tp_dictoffset */
    (initproc)PyOptimizerSGD_init, /* tp_init */
    0,                             /* tp_alloc */
    PyOptimizer_new,               /* tp_new */
    PyObject_GC_Del                /* tp_free */

};