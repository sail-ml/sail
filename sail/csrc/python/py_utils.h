#pragma once
#include <Python.h>

struct NoGIL {
    NoGIL() : save(PyEval_SaveThread()) {}
    ~NoGIL() { PyEval_RestoreThread(save); }

    PyThreadState* save;
};
