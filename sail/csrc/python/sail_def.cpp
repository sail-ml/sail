#define SAIL_CORE

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/dtypes.h"
// #include "core/modules/modules.h"
#include "numpy/arrayobject.h"

#include "error_defs.h"
#include "functions.h"
#include "initializers/initializers.h"
#include "py_dtypes/py_dtype.h"
#include "py_loss/py_loss.h"
#include "py_module/py_module.h"
#include "py_optimizer/py_optimizer.h"
#include "py_tensor/py_tensor.h"
#include "random/random_def.h"

#include "module_def.h"

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "sail_c", "Sail.", -1, 0};

static PyModuleDef random_module = {PyModuleDef_HEAD_INIT, "random",
                                    "Sail random Tensor generator.", -1, 0};

static PyModuleDef init_module = {PyModuleDef_HEAD_INIT, "init",
                                  "Sail Tensor initializers.", -1, 0};

static PyModuleDef loss_module = {PyModuleDef_HEAD_INIT, "loss",
                                  "Sail loss functions.", -1, 0};

static PyModuleDef optimizer_module = {PyModuleDef_HEAD_INIT, "optimizers",
                                       "Sail optimizers.", -1, 0};

PyMODINIT_FUNC PyInit_sail_init(void) {
    auto m = PyModule_Create(&init_module);
    PyModule_AddFunctions(m, InitFuncs);
    return m;
}

PyMODINIT_FUNC PyInit_sail_random(void) {
    auto m = PyModule_Create(&random_module);
    PyModule_AddFunctions(m, RandomFunctions);
    return m;
}

PyMODINIT_FUNC PyInit_sail_losses(void) {
    auto m = PyModule_Create(&loss_module);
    if (PyType_Ready(&PyLossType) < 0) return NULL;
    if (PyType_Ready(&PySCELossType) < 0) return NULL;
    if (PyType_Ready(&PyMSELossType) < 0) return NULL;

    if (m == NULL) return NULL;

    if (PyModule_AddObject(m, "Loss", (PyObject *)&PyLossType) < 0) {
        Py_DECREF(&PyLossType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "SoftmaxCrossEntropy",
                           (PyObject *)&PySCELossType) < 0) {
        Py_DECREF(&PySCELossType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "MeanSquaredError", (PyObject *)&PyMSELossType) <
        0) {
        Py_DECREF(&PyMSELossType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

PyMODINIT_FUNC PyInit_sail_optimizers(void) {
    auto m = PyModule_Create(&optimizer_module);

    if (PyType_Ready(&PyOptimizerType) < 0) return NULL;
    if (PyType_Ready(&PyOptimizerSGDType) < 0) return NULL;

    if (m == NULL) return NULL;

    if (PyModule_AddObject(m, "Optimizer", (PyObject *)&PyOptimizerType) < 0) {
        Py_DECREF(&PyOptimizerType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "SGD", (PyObject *)&PyOptimizerSGDType) < 0) {
        Py_DECREF(&PyOptimizerSGDType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

PyMODINIT_FUNC PyInit_libsail(void) {
    import_array();
    PyObject *m;
    if (PyType_Ready(&PyTensorType) < 0) return NULL;

    if (PyType_Ready(&PyDtypeBase) < 0) return NULL;

    m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    if (PyModule_AddObject(m, "Tensor", (PyObject *)&PyTensorType) < 0) {
        Py_DECREF(&PyTensorType);
        Py_DECREF(m);
        return NULL;
    }
    if (PyModule_AddObject(m, "dtype", (PyObject *)&PyDtypeBase) < 0) {
        Py_DECREF(&PyDtypeBase);
        Py_DECREF(m);
        return NULL;
    }

    PyObject *boolean = (PyObject *)generate_dtype(Dtype::sBool, 0);
    PyObject *int8 = (PyObject *)generate_dtype(Dtype::sInt8, 1);
    PyObject *uint8 = (PyObject *)generate_dtype(Dtype::sUInt8, 2);
    PyObject *int16 = (PyObject *)generate_dtype(Dtype::sInt16, 3);
    PyObject *uint16 = (PyObject *)generate_dtype(Dtype::sUInt16, 4);
    PyObject *int32 = (PyObject *)generate_dtype(Dtype::sInt32, 5);
    PyObject *uint32 = (PyObject *)generate_dtype(Dtype::sUInt32, 6);
    PyObject *int64 = (PyObject *)generate_dtype(Dtype::sInt64, 7);
    PyObject *uint64 = (PyObject *)generate_dtype(Dtype::sUInt64, 8);
    PyObject *float32 = (PyObject *)generate_dtype(Dtype::sFloat32, 11);
    PyObject *float64 = (PyObject *)generate_dtype(Dtype::sFloat64, 12);

    Py_INCREF(boolean);
    Py_INCREF(int8);
    Py_INCREF(uint8);
    Py_INCREF(int16);
    Py_INCREF(uint16);
    Py_INCREF(int32);
    Py_INCREF(uint32);
    Py_INCREF(int64);
    Py_INCREF(uint64);
    Py_INCREF(float32);
    Py_INCREF(float64);

    PyModule_AddObject(m, "DimensionError", PyDimensionError);
    PyModule_AddObject(m, "SailError", PySailError);

    PyModule_AddObject(m, "bool_", boolean);
    PyModule_AddObject(m, "int8", int8);
    PyModule_AddObject(m, "uint8", uint8);
    PyModule_AddObject(m, "int16", int16);
    PyModule_AddObject(m, "uint16", uint16);
    PyModule_AddObject(m, "int32", int32);
    PyModule_AddObject(m, "uint32", uint32);
    PyModule_AddObject(m, "int64", int64);
    PyModule_AddObject(m, "uint64", uint64);
    PyModule_AddObject(m, "float32", float32);
    PyModule_AddObject(m, "float64", float64);

    PyModule_AddFunctions(m, SailOpsMethods);

    PyObject *item;
    PyObject *sys_modules =
        PyImport_GetModuleDict();  // PyThreadState_GET()->interp->modules;

    PyModule_AddObject(m, "init", (item = PyInit_sail_init()));
    PyDict_SetItemString(sys_modules, "libsail.init", item);
    Py_INCREF(item);
    PyModule_AddObject(m, "random", (item = PyInit_sail_random()));
    PyDict_SetItemString(sys_modules, "libsail.random", item);
    Py_INCREF(item);
    PyModule_AddObject(m, "losses", (item = PyInit_sail_losses()));
    PyDict_SetItemString(sys_modules, "libsail.losses", item);
    Py_INCREF(item);
    PyModule_AddObject(m, "optimizers", (item = PyInit_sail_optimizers()));
    PyDict_SetItemString(sys_modules, "libsail.optimizers", item);
    Py_INCREF(item);
    PyModule_AddObject(m, "modules", (item = PyInit_sail_modules()));
    PyDict_SetItemString(sys_modules, "libsail.modules", item);
    Py_INCREF(item);
    // PyModule_AddObject(m, "modules", (item = PyInit_sail_losses()));
    // PyDict_SetItemString(sys_modules, "libsail.modules", item);
    // Py_INCREF(item);

    // PyObject* moduleDict = PyImport_GetModuleDict();
    // PyDict_SetItemString(moduleDict, "sail.csrc.libsail.init",
    //                      get_inits(m_init));

    /// RANDOM MODULE

    // create_capsule(m);

    return m;
}
#undef SAIL_CORE

// PyMODINIT_FUNC PyInit_libtest(void) {
//     import_py_tensor();
//     PyObject *m;
//     // if (PyType_Ready(&PyTensorType) < 0) return NULL;
//     m = PyModule_Create(&init_module2);
//     PyModule_AddFunctions(m, InitFuncs);
//     return m;
// }
