#pragma once

#include <Python.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "core/error.h"
// #include "core/modules/modules.h"
#include "numpy/arrayobject.h"

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
        return nullptr;
    }
    Py_DECREF(output);
    Py_DECREF(dict_globals);
    return d;
}
PyObject* exc_dict = make_getter_code();

PyObject* PySailError =
    PyErr_NewException("sail.SailError",
                       PyExc_Exception,  // use to pick base class
                       NULL);

PyObject* PyDimensionError =
    PyErr_NewException("sail.DimensionError",
                       PySailError,  // use to pick base class
                       NULL);

#define START_EXCEPTION_HANDLING try {
#define END_EXCEPTION_HANDLING                                        \
    }                                                                 \
    catch (const DimensionError& e) {                                 \
        PyErr_SetString(PyDimensionError, e.what());                  \
        return nullptr;                                               \
    }                                                                 \
    catch (const SailCError& e) {                                     \
        PyErr_SetString(PySailError, e.what());                       \
        return nullptr;                                               \
    }                                                                 \
    catch (const std::exception& e) {                                 \
        PyErr_SetString(PySailError,                                  \
                        ((std::string) "Unkown exception occured: " + \
                         (std::string)e.what())                       \
                            .c_str());                                \
        return nullptr;                                               \
    }

#define END_EXCEPTION_HANDLING_INT                                    \
    }                                                                 \
    catch (const DimensionError& e) {                                 \
        PyErr_SetString(PyDimensionError, e.what());                  \
        return -1;                                                    \
    }                                                                 \
    catch (const SailCError& e) {                                     \
        PyErr_SetString(PySailError, e.what());                       \
        return -1;                                                    \
    }                                                                 \
    catch (const std::exception& e) {                                 \
        PyErr_SetString(PySailError,                                  \
                        ((std::string) "Unkown exception occured: " + \
                         (std::string)e.what())                       \
                            .c_str());                                \
        return -1;                                                    \
    }\
