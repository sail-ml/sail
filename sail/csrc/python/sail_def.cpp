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
static PyModuleDef Mmodule = {PyModuleDef_HEAD_INIT, "modules",
                              "Example module that creates an extension type.",
                              -1, 0};

PyMODINIT_FUNC PyInit_libsail_c(void) {
    import_array();
    PyObject* m;
    if (PyType_Ready(&PyTensorType) < 0) return NULL;

    if (PyType_Ready(&PyDtypeInt32) < 0) return NULL;
    if (PyType_Ready(&PyDtypeFloat32) < 0) return NULL;
    if (PyType_Ready(&PyDtypeFloat64) < 0) return NULL;

    m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    Py_INCREF(&PyDtypeInt32);
    if (PyModule_AddObject(m, "Tensor", (PyObject*)&PyTensorType) < 0) {
        Py_DECREF(&PyTensorType);
        Py_DECREF(m);
        return NULL;
    }

    // PyObject* exc_dict = make_getter_code();

    // PyDimensionError = PyErr_NewException("sail.DimensionException",
    //                                       NULL,  // use to pick base class
    //                                       exc_dict);

    PyModule_AddObject(m, "DimensionError", PyDimensionError);
    PyModule_AddObject(m, "SailError", PySailError);
    // PyModule_AddObject(m, "modules", PyInit_modules());
    PyModule_AddFunctions(m, OpsMethods);

    PyModule_AddObject(m, "int32", (PyObject*)&PyDtypeInt32);
    PyModule_AddObject(m, "float32", (PyObject*)&PyDtypeFloat32);
    PyModule_AddObject(m, "float64", (PyObject*)&PyDtypeFloat64);

    /// RANDOM MODULE

    return m;
}
