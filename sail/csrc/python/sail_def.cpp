#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/dtypes.h"
// #include "core/modules/modules.h"
#include "numpy/arrayobject.h"

#include "error_defs.h"
#include "ops/ops_def.h"
#include "py_dtypes/py_dtype.h"
#include "py_module/py_module.h"
#include "py_tensor/py_tensor.h"

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "sail_c",
                             "Example module that creates an extension type.",
                             -1, 0};

PyMODINIT_FUNC PyInit_libsail_c(void) {
    import_array();
    PyObject* m;
    if (PyType_Ready(&PyTensorType) < 0) return NULL;

    if (PyType_Ready(&PyDtypeBase) < 0) return NULL;

    m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    if (PyModule_AddObject(m, "Tensor", (PyObject*)&PyTensorType) < 0) {
        Py_DECREF(&PyTensorType);
        Py_DECREF(m);
        return NULL;
    }

    PyObject* int32 = (PyObject *)generate_dtype(Dtype::sInt32, 5);
    PyObject* float32 = (PyObject *)generate_dtype(Dtype::sFloat32, 11);
    PyObject* float64 = (PyObject *)generate_dtype(Dtype::sFloat64, 12);

    Py_INCREF(int32);
    Py_INCREF(float32);
    Py_INCREF(float64);

    PyModule_AddObject(m, "DimensionError", PyDimensionError);
    PyModule_AddObject(m, "SailError", PySailError);
    PyModule_AddObject(m, "int32", int32);
    PyModule_AddObject(m, "float32", float32);
    PyModule_AddObject(m, "float64", float64);
    PyModule_AddFunctions(m, OpsMethods);


    /// RANDOM MODULE

    return m;
}
