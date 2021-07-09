import yaml
from .utils import *

def run(input, output):
    with open(input, 'r') as stream:
        try:
            functions = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)



    FUNCTION_CODE = """
    PyObject* sail_{function_name}(PyObject* self, PyObject* args, PyObject* kwargs) {{
        START_EXCEPTION_HANDLING
        {dispatches}
        END_EXCEPTION_HANDLING
    }}
    """
    NEW_FUNCTION_CODE = """
    PyObject* sail_{function_name}(PyObject* self, PyObject* args, PyObject* kwargs) {{
        START_EXCEPTION_HANDLING
        {arg_parser}
        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);
        {dispatches}
        {parent_set}
        return (PyObject*)ret_class;
        END_EXCEPTION_HANDLING
    }}
    """

    PARENT_SET = """
    ret_class->base_object = (PyObject*)parser.py_tensor({idx});
    Py_INCREF(ret_class->base_object);
    """

    ARG_PARSER = """
    PythonArgParser<{N}> parser = PythonArgParser<{N}>(
        {{
            {sigs}
        }}, args, kwargs);

    parser.parse();
    
    """

    DISPATCHER = """
    if (parser.at({idx})) {{
        ret_class->tensor = {internal_call};
    }}
    """

    REDUCTION_DISPATCH_CODE = """
        PyTensor* t1;
        PyObject* axis = Py_None;
        int keepdims = 0;
        static char* kwlist[] = {{"tensor", "axis", "keepdims", NULL}};

        REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims);

        PyTensor* ret_class;
        ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

        if (axis == Py_None) {{
            ret_class->tensor =
                {internal_func}(((PyTensor*)t1)->tensor, NULLDIM, (bool)keepdims);
        }} else {{
            ret_class->tensor = {internal_func}(((PyTensor*)t1)->tensor,
                                            PyLong_AsLong(axis), (bool)keepdims);
        }}

        return (PyObject*)ret_class;
    """

    DISPATCH_CODE = """
        {variables}
        static char* kwlist[] = {{ {names}, NULL }};
        if (PyArg_ParseTupleAndKeywords(args, kwargs, "{codes}", kwlist, {parse_args})) {{
            PyTensor* ret_class;
            ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

            ret_class->tensor = {internal_func}({args});
            return (PyObject*)ret_class;
        }}"""

    CUSTOM_DISPATCH_CODE = """
        {variables}
        static char* kwlist[] = {{ {names}, NULL }};
        if (PyArg_ParseTupleAndKeywords(args, kwargs, "{codes}", kwlist, {parse_args})) {{
            PyTensor* ret_class;
            ret_class = (PyTensor*)PyTensorType.tp_alloc(&PyTensorType, 0);

            {custom}

            return (PyObject*)ret_class;
        }}"""

    ERROR_CODE = """
        else {
            PyErr_SetString(PyExc_TypeError, "incorrect arguments");
            return nullptr;
        }
    """

    FUNCS_LIST = """
    static PyMethodDef SailOpsMethods[] = {{
        {funcs}
        {{NULL}} }};
    """

    FUNCS_MEMBER = """{{"{name}", (PyCFunction)sail_{name}, METH_VARARGS | METH_KEYWORDS, NULL}},"""

    funcs = []
    used_funcs = []
    for f in functions:

        function_name = f 

        if ("new" in functions[f]):
            sigs = functions[f]["signatures"]
            calls = functions[f]["internal_calls"]
            sigs = ['"' + s + '"' for s in sigs]
            n = 0
            for s in sigs:
                c = len(s.split(","))
                if c > n:
                    n = c
            ap = ARG_PARSER.format(N=n, sigs=",\n".join(sigs))
            calls_ = []
            i = 0
            for c in calls:
                calls_.append(DISPATCHER.format(idx=i, internal_call=c))
                i += 1

            calls = " else ".join(calls_)
            if ("set_base" in functions[f]):
                parent = PARENT_SET.format(idx=functions[f]["set_base"]["index"])
            else:
                parent = ""
            funcs.append(NEW_FUNCTION_CODE.format(function_name=function_name, arg_parser=ap, dispatches=calls, parent_set=parent))
            used_funcs.append(FUNCS_MEMBER.format(name=function_name))
            continue

        dispatches = []
        if "full_impl" in functions[f]:
            funcs.append(functions[f]["full_impl"])
            used_funcs.append(FUNCS_MEMBER.format(name=function_name))
            continue
        
        if ("special_impl" in functions[f]):
            if functions[f]["special_impl"] == "reduction":
                dispatches = REDUCTION_DISPATCH_CODE.format(
                    internal_func=functions[f]["internal_func"])
        else:
            cd = False
            if ("custom_dispatch" in functions[f]):
                cd = True
                assert len(functions[f]["signatures"]) == len(functions[f]["custom_dispatch"])

            z = 0
            for s in functions[f]["signatures"]:
                d = process_dispatch(functions[f]["internal_func"], s)
                if cd:
                    d["custom"] = functions[f]["custom_dispatch"][z]
                    z += 1
                    d = CUSTOM_DISPATCH_CODE.format(**d)
                else:
                    d = DISPATCH_CODE.format(**d)
                dispatches.append(d)

            dispatches = "\nelse ".join(dispatches)
            dispatches += "\n" + ERROR_CODE

        
        funcs.append(FUNCTION_CODE.format(function_name=function_name, dispatches=dispatches))
        used_funcs.append(FUNCS_MEMBER.format(name=function_name))


    file = """
    #pragma once

    #include <Python.h>
    #include <structmember.h>
    #include <algorithm>
    #include <chrono>
    #include <iostream>
    #include <vector>
    #include "py_tensor/py_tensor.h"
    #include "core/Tensor.h"
    #include "core/ops/ops.h"
    #include "core/tensor_shape.h"
    #include "numpy/arrayobject.h"
    #include "arg_parser.h"


    #define REDUCATION_ARGS(args, kwargs, kwlist, t1, axis, keepdims)           \\
        {                                                                       \\
            if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist, &t1, \\
                                            &axis, &keepdims)) {               \\
                PyErr_SetString(PyExc_TypeError,                                \\
                                "must pass a tensor and an integer for axis");  \\
                return nullptr;                                                 \\
            }                                                                   \\
        }


    """

    file += "\n".join(funcs)

    ops_methods = FUNCS_LIST.format(funcs="\n    ".join(used_funcs))

    file += ops_methods

    with open(output, "w") as f:
        f.write(file)
