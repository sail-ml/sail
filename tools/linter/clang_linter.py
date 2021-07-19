import os, subprocess
import glob, json, yaml
import time, signal
import random
from lint_utils import get_files

def get_default_gcc_includes():

    with open("t.c", "w") as f:
        f.write("#include <bogus.h>")
    try:
        cmd = subprocess.check_output(['gcc', '-v', 't.c'], stderr=subprocess.STDOUT)
    except Exception as e:
        output = str(e.output.decode())
    subprocess.run(["rm", "-rf", "t.c"])

    start = output.find('#include <...>', 0, len(output))
    end = output.find('End of search list', 0, len(output))

    focus = output[start:end].split("\n")[1:]
    focus = [f.strip() for f in focus][:-1]

    return focus

def clean_an_show(string):
    out = []
    stop = False
    for line in string.split("\n"):
        if ("generated." in line):
            stop = True
        elif ("in non-user code" in line):
            stop = True 
        elif ("Use -system-headers to display errors" in line):
            stop = True
        elif (line.startswith("/") or line.startswith("../")):
            if ("libs/xsimd/include" in line or "c++" in line):
                stop = True
            else:
                stop = False

        if (not stop):
            out.append(line)

    if (out != []):
        print ("\n".join(out))
        return False 
    return True

def launch_clang(command, file):
    command = list(command)
    command.append(file)

    x = subprocess.check_output(command, stderr=subprocess.STDOUT).decode().strip() 
    return clean_an_show(x)

def launch2(command):
    command = list(command)

    x = subprocess.check_output(command, stderr=subprocess.STDOUT).decode().strip() 
    clean_an_show(x)

def check_file(f):
    if (f.endswith(".cpp") or f.endswith(".cc") or f.endswith(".c")):
        return True
    return False

def execute_clang(args):

    config_file = args.config_file

    with open(config_file) as config_:
        config = json.dumps(yaml.load(config_, Loader=yaml.SafeLoader))

    command = ["clang-tidy", "-config", config, "-extra-arg=-std=c++17", "-p", "build/temp.linux-x86_64-3.7"]

    if (args.verbose):
        print (" ".join(command))

    files = get_files(args)
    # print ("linting the following files:")
    res = True 
    for file in files:
        print ("Linting " + file)   
        r = (launch_clang(command, file))
        

    # command += files

    # launch2(command)


  

