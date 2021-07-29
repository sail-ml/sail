import yaml, argparse, re, os


blacklist_ops = [
   "add_docstring",
]

def convert_type(type):
    if (type == "IntList"):
        return "List[int]"
    if (type == "TensorList"):
        return "List[Tensor]"
    if (type == "ListOfIntList"):
        return "List[List[int]]"
    return type 

def convert_value(v):
    if v == "false":
        return False 
    if v == "true":
        return True
    return v

def convert_args_to_pyi(args):
    nargs = []
    for a in args:
        a = a.strip().split(" ")
        if (len(a) == 2):
            nargs.append([a[1], convert_type(a[0])])
        else:
            nargs.append([a[1], convert_type(a[0]), convert_value(a[3])])

    return nargs

def generate_stubs(func_yml, path):
    with open(func_yml, 'r') as stream:
        try:
            funcs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if (os.path.exists(path + ".psrc")):
        with open(path + ".psrc", "r") as f:
            file = f.read()
    else:
        file = """from typing import Any, ClassVar
from typing import overload
from typing import List, Set, Dict, Tuple, Optional

import numpy as np 
import sail"""


    stubs = {}
    
    for o in funcs:
        if (o in blacklist_ops):
            continue   
        op_hints = []
        function_code = funcs[o]
        sigs = function_code["signatures"]

        for s in sigs:
            args = re.sub(".*\(", "(", s)[1:-1].split(", ")
            converted_args = convert_args_to_pyi(args)
            arg_string = []
            for a in converted_args:
                if (len(a) == 2):
                    arg_string.append("%s: %s" % (a[0], a[1]))
                else:
                    arg_string.append("%s: %s = %s" % (a[0], a[1], a[2]))
            sig_code = ("def {}(" + ", ".join(arg_string) + ") -> " + function_code["return"] + ": ...").format(o)
            op_hints.append(sig_code)

        stubs[o] = op_hints 

    funcs = ""
    for s in stubs:
        s = stubs[s]
        if (len(s) == 1):
            funcs += s[0] + "\n"
        else:
            for i in s:
                funcs += "@overload\n" + i + "\n"
        funcs += "\n"

    with open(path, "w") as f:
        f.write(file.format(funcs=funcs))    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate stubs for Sail')
    parser.add_argument('--functions-path', metavar='NATIVE',
                        default='sail/csrc/python/generate_utils/functions.yaml',
                        help='path to function.yaml')
    parser.add_argument('--out', metavar='OUT',
                        default='sail/csrc/libsail.pyi',
                        help='path to output directory')
    args = parser.parse_args()
    generate_stubs(args.functions_path, args.out)