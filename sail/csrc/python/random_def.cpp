#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/dtypes.h"
// #include "core/modules/modules.h"
#include "numpy/arrayobject.h"

#include "error_defs.h"
#include "py_dtypes/py_dtype.h"
#include "py_module/py_module.h"
#include "py_tensor/py_tensor.h"
#include "random/random_def.h"

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "random",
                             "Example module that creates an extension type.",
                             -1, 0};

PyMODINIT_FUNC PyInit_librandom(void) {
    import_array();
    PyObject* m;

    if (PyType_Ready(&PyTensorType) < 0) return NULL;

    m = PyModule_Create(&module);

    if (m == NULL) return NULL;
    PyModule_AddFunctions(m, RandomFunctions);

    return m;
}
