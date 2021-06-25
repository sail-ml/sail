#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/dtypes.h"
// #include "core/modules/modules.h"
#include "numpy/arrayobject.h"

#include "py_dtypes/py_dtype.h"
#include "py_module/py_module.h"
#include "py_tensor/py_tensor.h"

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "modules",
                             "Modules for SAIL", -1, 0};

PyMODINIT_FUNC PyInit_libmodules(void) {
    import_array();
    PyObject* m;

    if (PyType_Ready(&PyModuleType) < 0) return NULL;
    if (PyType_Ready(&PyLinearModuleType) < 0) return NULL;
    if (PyType_Ready(&PySigmoidModuleType) < 0) return NULL;
    if (PyType_Ready(&PySoftmaxModuleType) < 0) return NULL;
    if (PyType_Ready(&PyReLUModuleType) < 0) return NULL;
    if (PyType_Ready(&PyTensorType) < 0) return NULL;

    m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    if (PyModule_AddObject(m, "Module", (PyObject*)&PyModuleType) < 0) {
        Py_DECREF(&PyModuleType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "Linear", (PyObject*)&PyLinearModuleType) < 0) {
        Py_DECREF(&PyLinearModuleType);
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "Sigmoid", (PyObject*)&PySigmoidModuleType) < 0) {
        Py_DECREF(&PySigmoidModuleType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "Softmax", (PyObject*)&PySoftmaxModuleType) < 0) {
        Py_DECREF(&PySoftmaxModuleType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "ReLU", (PyObject*)&PyReLUModuleType) < 0) {
        Py_DECREF(&PyReLUModuleType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
