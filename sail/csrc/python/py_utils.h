#pragma once
#include <Python.h>
#include "core/Tensor.h"
#include "py_tensor/py_tensor.h"

struct NoGIL {
    NoGIL() : save(PyEval_SaveThread()) {}
    ~NoGIL() { PyEval_RestoreThread(save); }

    PyThreadState* save;
};

sail::Tensor unpack_pytensor(PyObject* t) { return ((PyTensor*)t)->tensor; }
