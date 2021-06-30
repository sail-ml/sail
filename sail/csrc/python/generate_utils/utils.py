
def convert_type_to_ctype(t):
    if (t == "Tensor" or t == "PyTensor"):
        return "PyTensor *"
    elif (t == "sequence"):
        return "PyObject *"
    elif (t == "double"):
        return "double"
    elif (t == "int"):
        return "int"
    elif (t == "bool"):
        return "bool"

def convert_type_to_pycode(t):
    if (t == "Tensor" or t == "PyTensor"):
        return "O"
    elif (t == "sequence"):
        return "O"
    elif (t == "double"):
        return "d"
    elif (t == "int"):
        return "i"
    elif (t == "bool"):
        return "b"

def convert_type_to_arg(t, a):
    if (t == "Tensor" or t == "PyTensor"):
        return "%s->tensor" % a
    elif (t == "sequence"):
        return "seq_%s" % a
    elif (t == "double"):
        return a
    elif (t == "int"):
        return a
    elif (t == "bool"):
        return a
        
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
    return {"basic_args": ", ".join(variable_names), "variables": variables, "names": names, "codes": codes,
    "parse_args": parse_args, "internal_func": internal_func, "args":args}