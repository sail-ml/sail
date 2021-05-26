#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "../src/Tensor.h"
#include "../src/dtypes.h"
#include "numpy/arrayobject.h"

#include "ops/ops_def.h"
#include "py_dtypes/py_dtype.h"
#include "py_tensor/py_tensor.h"

PyObject* make_getter_code() {
    const char* code =
        "def code(self):\n"
        "  try:\n"
        "    return self.args[1]\n"
        "  except IndexError:\n"
        "    return -1\n"
        "code = property(code)\n"
        "def message(self):\n"
        "  try:\n"
        "    return self.args[0]\n"
        "  except IndexError:\n"
        "    return ''\n"
        "\n";

    PyObject* d = PyDict_New();
    PyObject* dict_globals = PyDict_New();
    PyDict_SetItemString(dict_globals, "__builtins__", PyEval_GetBuiltins());
    PyObject* output = PyRun_String(code, Py_file_input, dict_globals, d);
    if (output == NULL) {
        Py_DECREF(d);
        return NULL;
    }
    Py_DECREF(output);
    Py_DECREF(dict_globals);
    return d;
}

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "sail_c",
                             "Example module that creates an extension type.",
                             -1, 0};

PyMODINIT_FUNC PyInit_random(void) {
    import_array();
    PyObject* m;

    m = PyModule_Create(&module);
    if (m == NULL) return NULL;
    PyModule_AddFunctions(m, RandomFactories);

    return m;
}

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

    PyObject* exc_dict = make_getter_code();

    PyDimensionError = PyErr_NewException("sail.DimensionException",
                                          NULL,  // use to pick base class
                                          exc_dict);

    PyModule_AddObject(m, "DimensionError", PyDimensionError);
    PyModule_AddObject(m, "random", PyInit_random());
    PyModule_AddFunctions(m, OpsMethods);

    PyModule_AddObject(m, "int32", (PyObject*)&PyDtypeInt32);
    PyModule_AddObject(m, "float32", (PyObject*)&PyDtypeFloat32);
    PyModule_AddObject(m, "float64", (PyObject*)&PyDtypeFloat64);

    /// RANDOM MODULE

    return m;
}
