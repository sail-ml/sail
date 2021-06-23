import yaml

with open("functions.yaml", 'r') as stream:
    try:
        functions = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def convert_type_to_ctype(t):
    if (t == "Tensor" or t == "PyTensor"):
        return "PyTensor *"
    elif (t == "sequence"):
        return "PyObject *"
    elif (t == "double"):
        return "double"
    elif (t == "int"):
        return "int"

def convert_type_to_pycode(t):
    if (t == "Tensor" or t == "PyTensor"):
        return "O"
    elif (t == "sequence"):
        return "O"
    elif (t == "double"):
        return "d"
    elif (t == "int"):
        return "i"

def convert_type_to_arg(t, a):
    if (t == "Tensor" or t == "PyTensor"):
        return "%s->tensor" % a
    elif (t == "sequence"):
        return "seq_%s" % a
    elif (t == "double"):
        return a
    elif (t == "int"):
        return a


FUNCTION_CODE = """
PyObject* sail_{function_name}(PyObject* self, PyObject* args, PyObject* kwargs) {{
    START_EXCEPTION_HANDLING
    {dispatches}
    END_EXCEPTION_HANDLING
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
def process_dispatch(internal_func, signature):


    signature_args = signature.split("(")[1].split(")")[0].split(", ")
    variable_defs = []
    variable_names = []
    required_parse_codes = []
    optional_parse_codes = []
    args = []
    for s in signature_args:        
        vdef = []
        vdef2 = ""
        default = False
        if "=" in s:
            default = True
        
        arg_data = s.split(" ")
        type_ = arg_data[0]
        vdef2 = convert_type_to_ctype(type_)
        if (not default):
            vdef2 += " " + arg_data[1] + ";"
            variable_names.append(arg_data[1])
            args.append(convert_type_to_arg(type_, arg_data[1]))
            if (optional_parse_codes != []):
                raise Exception(signature + " is invalid")
            required_parse_codes.append(convert_type_to_pycode(type_))
        else:
            d = arg_data[1].split("=")
            if (arg_data[1] == "None"):
                arg_data[1] = "NULL"
            
            vdef2 += " " + d[0] + " = " + d[1] + ";"
            variable_names.append(d[0])
            args.append(convert_type_to_arg(type_, d[0]))

            optional_parse_codes.append(convert_type_to_pycode(type_))

        variable_defs.append(vdef2)

    if (optional_parse_codes != []):
        codes = "".join(required_parse_codes) + "|" + "".join(optional_parse_codes)
    else:
        codes = "".join(required_parse_codes)
    

    names = ", ".join(['"%s"' % n for n in variable_names])
    parse_args = ", ".join(["&%s" % n for n in variable_names])
    args = ", ".join(args)
    variables = "\n    ".join(variable_defs)

    # return DISPATCH_CODE.format(variables=variables, names=names, codes=codes, parse_args=parse_args, internal_func=internal_func, args=args)
    return {"variables": variables, "names": names, "codes": codes,
    "parse_args": parse_args, "internal_func": internal_func, "args":args}

        

used_funcs = []
for f in functions:

    function_name = f 
    dispatches = []
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

with open("functions.h", "w") as f:
    f.write(file)
