from clang_linter import launch_clang
from custom_linter import launch_custom
from lint_utils import get_files
import json, yaml

def execute(args):

    config_file = args.config_file

    with open(config_file) as config_:
        config = json.dumps(yaml.load(config_, Loader=yaml.SafeLoader))

    command = ["clang-tidy", "-config", config, "-extra-arg=-std=c++17", "-p", "build/temp.linux-x86_64-3.7"]

    if (args.verbose):
        print (" ".join(command))

    files = get_files(args, [".c", ".cc", ".cpp", ".h", ".hpp"])

    for file in files:
        print ("Linting " + file)  
        res = True
        if (not args.no_clang and ".h" not in file):
            res = (launch_clang(command, file) and res)
            print (res)
        if (not args.no_custom):
            res = (launch_custom(file) and res)
    print (res)
    # exit(0)
    if (not res):
        exit(1)