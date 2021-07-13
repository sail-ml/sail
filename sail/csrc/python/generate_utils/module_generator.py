import yaml
from .utils import *

FULL_DEF = """
{init}

{forward}

{getsets}

{pygetset}
{pyfuncdef}

{typedef}

"""

READY = """ 
    if (PyType_Ready(&Py{name}ModuleType) < 0) return NULL;
"""

ADD_OBJECT = """
if (PyModule_AddObject(m, "{name}", (PyObject*)&Py{name}ModuleType) < 0) {{
        Py_DECREF(&Py{name}ModuleType);
        Py_DECREF(m);
        return NULL;
    }}
"""

INIT_CODE = """
static int Py{name}Module_init(PyModule *self, PyObject *args,
                               PyObject *kwargs) {{
    START_EXCEPTION_HANDLING
    {variables}
    static char* kwlist[] = {{ {names}, NULL }};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "{codes}", kwlist, {parse_args})) {{
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return -1;
    }}

    self->module = (Module *)(new sail::modules::{name}({basic_args}));
    return 0;
    END_EXCEPTION_HANDLING_INT
}}
"""
CUSTOM_INIT_CODE = """
static int Py{name}Module_init(PyModule *self, PyObject *args,
                               PyObject *kwargs) {{
    START_EXCEPTION_HANDLING
    {code}
    END_EXCEPTION_HANDLING_INT
}}
"""
EMPTY_INIT_CODE = """
static int Py{name}Module_init(PyModule *self, PyObject *args,
                               PyObject *kwargs) {{
    START_EXCEPTION_HANDLING

    self->module = (Module *)(new sail::modules::{name}());
    return 0;
    END_EXCEPTION_HANDLING_INT
}}
"""
FORWARD_CODE = """
static PyObject* Py{name}Module_forward(PyModule *self, PyObject *args,
                               PyObject *kwargs) {{
    START_EXCEPTION_HANDLING
    {variables}
    static char* kwlist[] = {{ {names}, NULL }};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "{codes}", kwlist, {parse_args})) {{
        PyErr_SetString(PyExc_TypeError, "incorrect arguments");
        return -1;
    }}
    PyTensor *py_output = (PyTensor *)PyTensorType.tp_alloc(&PyTensorType, 0);

    sail::Tensor output = ((sail::modules::{name} *)(self->module))->forward({call_args});

    GENERATE_FROM_TENSOR(py_output, output);
    return (PyObject *)py_output;
    END_EXCEPTION_HANDLING
}}
"""

FUNC_LIST = """ 
static PyMethodDef Py{name}Module_methods[] = {{
    {{"forward", (PyCFunction)Py{name}Module_forward,
     METH_VARARGS | METH_KEYWORDS, NULL}},
    {{NULL}} /* Sentinel */
}};
"""

TENSOR_GETTER_CODE = """
static PyObject *Py{name}Module_get_{property}(PyModule *self, void* closure) {{
    START_EXCEPTION_HANDLING
    PyTensor* x = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);
    sail::Tensor xx = (*(sail::modules::{name} *)(self->module)).{property};
    GENERATE_FROM_TENSOR(x, xx);
    return (PyObject *)x;
    END_EXCEPTION_HANDLING
}}
"""
TENSOR_SETTER_CODE = """
static int Py{name}Module_set_{property}(PyModule *self, PyTensor *t,
                                      void *closure) {{
    START_EXCEPTION_HANDLING
    ((sail::modules::{name} *)(self->module))->set_{property}(t->tensor);
    return 0;
    END_EXCEPTION_HANDLING_INT
}}
"""
IGNORE_SETTER_CODE = """
static int Py{name}Module_set_{property}(PyModule *self, PyObject *t,
                                      void *closure) {
    PyErr_SetString(PyExc_AttributeError, "attribute cannot be set");
    return -1;
}
"""

PYGETSET = """ 
static PyGetSetDef Py{name}Module_get_setters[] = {{
    {funcs}
    {{NULL}}
}};
"""
PYGETSET_INNER = """ 
{{"{property}", (getter)Py{name}Module_get_{property}, (setter)Py{name}Module_set_{property}, NULL}},
"""

TYPE_DEF = """
static PyTypeObject Py{name}ModuleType = {{
    PyVarObject_HEAD_INIT(NULL, 0) "sail.{name}", /* tp_name */
    sizeof(PyModule),                                  /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)PyModule_dealloc,                      /* tp_dealloc */
    0,                                                 /* tp_print */
    0,                                                 /* tp_getattr */
    0,                                                 /* tp_setattr */
    0,                                                 /* tp_reserved */
    0,                                                 /* tp_repr */
    0,                                                 /* tp_as_number */
    0,                                                 /* tp_as_sequence */
    0,                                                 /* tp_as_mapping */
    0,                                                 /* tp_hash */
    Py{name}Module_forward,                            /* tp_call */
    0,                                                 /* tp_str */
    0,                                                 /* tp_getattro */
    0,                                                 /* tp_setattro */
    0,                                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC,          /* tp_flags */
    NULL,                            /* tp_doc */
    (traverseproc)PyModule_traverse, /* tp_traverse */
    (inquiry)PyModule_clear,         /* tp_clear */
    0,                               /* tp_richcompare */
    0,                               /* tp_weaklistoffset */
    0,                               /* tp_iter */
    0,                               /* tp_iternext */
    Py{name}Module_methods,          /* tp_methods */
    0,                               /* tp_members */
    Py{name}Module_get_setters,      // PyModule_getsetters, /* tp_getset */
    &PyModuleType,                   /* tp_base */
    0,                               /* tp_dict */
    0,                               /* tp_descr_get */
    0,                               /* tp_descr_set */
    0,                               /* tp_dictoffset */
    (initproc)Py{name}Module_init,   /* tp_init */
    0,                               /* tp_alloc */
    PyModule_new,                    /* tp_new */
    PyObject_GC_Del                  /* tp_free */

}};
"""

def create_init(name, init):
    if "custom" not in init:
        sig = init["signature"]
        if (len(sig) == 2):
            return EMPTY_INIT_CODE.format(name=name)
        args = process_dispatch(name, sig)
        return INIT_CODE.format(name=name, variables=args["variables"], names=args["names"],
                codes=args["codes"], parse_args=args["parse_args"], basic_args=args["basic_args"])
    else:
        return CUSTOM_INIT_CODE.format(name=name, code=init["custom"])
def create_forward(name, sig):
    sig = sig["signature"]
    args = process_dispatch(name, sig)
    return FORWARD_CODE.format(name=name, variables=args["variables"], names=args["names"],
            codes=args["codes"], call_args=args["args"], parse_args=args["parse_args"])


def run(input, output, output2):
    with open(input, 'r') as stream:
        try:
            modules = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    mod_defs = []
    readies = []
    adds = []
    for m in modules:
        module_meta = modules[m]
        init_def = create_init(m, module_meta["init"])
        if ("properties" in module_meta):
            props = module_meta["properties"]
        else:
            props = []
        getsets = []
        getset_inner = []
        for p in props:
            if (props[p]["type"] == "Tensor"):
                gets = TENSOR_GETTER_CODE.format(name=m, property=p)
                if (props[p]["write"]):
                    sets = TENSOR_SETTER_CODE.format(name=m, property=p)
                else:
                    sets = IGNORE_SETTER_CODE.format(name=m, property=p)
                getsets.append("\n".join([gets, sets]))
                getset_inner.append(PYGETSET_INNER.format(property=p, name=m))

        forward_func = create_forward(m, module_meta["forward"])
        td = TYPE_DEF.format(name=m)
        fl = FUNC_LIST.format(name=m)

        x = FULL_DEF.format(forward=forward_func, init=init_def, getsets="\n".join(getsets), 
                            pygetset=PYGETSET.format(name=m, funcs="\n".join(getset_inner)), 
                            typedef=td, pyfuncdef=fl)
        mod_defs.append(x)
        adds.append(ADD_OBJECT.format(name=m))
        readies.append(READY.format(name=m))

   


    file = """
    #pragma once
    #include "../error_defs.h"
    #include "../macros.h"
#include "../arg_parser.h"
    #include "../py_tensor/py_tensor_def.h"
    #include "core/modules/modules.h"

    #include "py_module_def.h"


    """

    file += "\n".join(mod_defs)

    moddef = """ 
    #pragma once
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



PyObject* get_modules(PyObject* m) {{

    if (PyType_Ready(&PyModuleType) < 0) return NULL;
    {readies}

    if (m == NULL) return NULL;

    if (PyModule_AddObject(m, "Module", (PyObject*)&PyModuleType) < 0) {{
        Py_DECREF(&PyModuleType);
        Py_DECREF(m);
        return NULL;
    }}
    {adds}

    return m;
}}

    """

    file2 = moddef.format(readies="\n".join(readies), adds="\n".join(adds))

    with open(output, "w") as f:
        f.write(file)
    with open(output2, "w") as f:
        f.write(file2)
