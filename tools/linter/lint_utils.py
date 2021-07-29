import os 
import glob 

def check_against_whitelist(file, whitelist):
    for e in whitelist:
        if (file.endswith(e)):
            return True
    return False

def get_files(args, whitelist=[".c", ".cpp", ".cc"], throw=True):
    if (args.include):
        if (check_against_whitelist(args.include, whitelist)):
            return [args.include]
        elif (throw):
            raise Exception("File type is blacklisted. Type: .%s, Whitelist: %s" % (args.include.split(".")[-1], whitelist))

    elif (args.diff_file):
        files = []
        with open(args.diff_file) as f:
            for file in f.read().split("\n"):
                if (check_against_whitelist(file, whitelist)):
                    files.append(file)
        return files
    else:
        files = []
        base_path = args.base_path 
        for w in whitelist:
            files += list(glob.glob(os.path.join(base_path, "**/*%s" % w), recursive=True))
        return files