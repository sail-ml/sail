#include <Python.h>
#include <core/Tensor.h>
#include <core/dtypes.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "numpy/arrayobject.h"

#include "ops/ops_def.h"
#include "py_dtypes/py_dtype.h"
#include "py_loss/py_loss.h"
#include "py_tensor/py_tensor.h"

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "loss",
                             "Example module that creates an extension type.",
                             -1, 0};

PyMODINIT_FUNC PyInit_liblosses(void) {
    import_array();
    PyObject* m;

    if (PyType_Ready(&PyLossType) < 0) return NULL;
    if (PyType_Ready(&PySCELossType) < 0) return NULL;
    if (PyType_Ready(&PyMSELossType) < 0) return NULL;
    if (PyType_Ready(&PyTensorType) < 0) return NULL;

    m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    if (PyModule_AddObject(m, "Loss", (PyObject*)&PyLossType) < 0) {
        Py_DECREF(&PyLossType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "SoftmaxCrossEntropy",
                           (PyObject*)&PySCELossType) < 0) {
        Py_DECREF(&PySCELossType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "MeanSquaredError", (PyObject*)&PyMSELossType) <
        0) {
        Py_DECREF(&PyMSELossType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
