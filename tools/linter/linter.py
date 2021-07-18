import os, subprocess
import glob, json, yaml
import time, signal
import random

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
        if (line.startswith("/") or line.startswith("../")):
            if ("libs/xsimd/include" in line or "c++" in line):
                stop = True
            else:
                stop = False
        
        if (not stop):
            out.append(line)

    print ("\n".join(out))

def launch(command, file):
    command = list(command)
    command.append(file)

    x = subprocess.check_output(command).decode().strip() 
    # print (x)
    clean_an_show(x)
def launch2(command):
    command = list(command)

    x = subprocess.check_output(command).decode().strip() 
    # print (x)
    clean_an_show(x)

def execute(args):

    config_file = args.config_file

    with open(config_file) as config_:
        config = json.dumps(yaml.load(config_, Loader=yaml.SafeLoader))

    command = ["clang-tidy", "-config", config, "-extra-arg=-std=c++17", "-p", "build/temp.linux-x86_64-3.7"]


    base_path = args.base_path 

    if (args.verbose):
        print (" ".join(command))


    if (args.include):
        if (str(args.include).endswith(".h")):
            raise Exception("Cannot lint header files")
        command.append(args.include)
        launch(command, args.include)
    
    else:
        base_path_source = glob.glob(os.path.join(base_path, "**/*.cpp"), recursive=True)

        for file in base_path_source:
            # command.append(file)

            launch(command, file)


  

