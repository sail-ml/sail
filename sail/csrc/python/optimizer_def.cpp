#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/dtypes.h"
// #include "core/modules/modules.h"
#include "numpy/arrayobject.h"

#include "py_dtypes/py_dtype.h"
#include "py_optimizer/py_optimizer.h"
#include "py_tensor/py_tensor.h"

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "optimizers",
                             "Example module that creates an extension type.",
                             -1, 0};

PyMODINIT_FUNC PyInit_liboptimizers(void) {
    import_array();
    PyObject* m;

    if (PyType_Ready(&PyOptimizerType) < 0) return NULL;
    if (PyType_Ready(&PyOptimizerSGDType) < 0) return NULL;
    if (PyType_Ready(&PyTensorType) < 0) return NULL;

    m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    if (PyModule_AddObject(m, "Optimizer", (PyObject*)&PyOptimizerType) < 0) {
        Py_DECREF(&PyOptimizerType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "SGD", (PyObject*)&PyOptimizerSGDType) < 0) {
        Py_DECREF(&PyOptimizerSGDType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
