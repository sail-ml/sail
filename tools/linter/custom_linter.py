from enum import Enum
import re, os

MAX_FCN = 65

FUNCTION_REGEX_SOURCE = "\s*(inline|)\s*\w*(::\w*(<\w*>|)|)(&|)(\*|) \w*(::\w*(<\w*>|)|)([\[\]<>+=\-\*\/]*|)\([a-zA-Z0-9_\*<>&:, ]*\) (const|) {"
ACCEPT_COMMENT_REGEX = "\s*\/\/[\s[A-Za-z0-9,\.-]*]*$"
CLASS_FIND_REGEX = "class \w* {"


ALLOW_NO_HEADER = "allow-no-header"
ALLOW_NO_SOURCE = "allow-no-source"
ALLOW_CLASS_DEFS = "allow-class-definitions"
ALLOW_HEADER_IMPL = "allow-impl-in-header"

ERROR_BASE = "Error: {file}"
WARNING_BASE = "Warning: {file}"

class Error(Enum):
    COMMENT = 1
    CLASS_IN_SOURCE = 2
    FUNCTION_TOO_LONG = 3
    FUNCTION_IN_HEADER = 4

class ErrorText(Enum):
    CODE_WITH_COMMENT = ERROR_BASE + ":(Line {line}) Comments containing code must be deleted"
    CLASS_DEF_IN_SOURCE = ERROR_BASE + ":(Line {line}) Class definitions should be in corresponding headers"
    FUNCTION_TOO_LONG = ERROR_BASE + """ Function spans too many lines. Recommended to be less than """ + str(MAX_FCN) + """ (currently {line_count})
\tLine {line} | {code} """
    FUNCTION_IN_HEADER = ERROR_BASE + ":(Line {line}) Function implementations should not be in header code"
    HEADER_FILE_MISSING = ERROR_BASE + " Source files must have corresponding header"
    SOURCE_FILE_MISSING = ERROR_BASE + " Header files must have corresponding source file"

    def format(text, **kwargs):
        return text.value.format(**kwargs)

    def __str__(self):
        return self.value 


header_endings = [".h", ".hpp", ".hpp.src"]
source_endings = [".c", ".cpp", ".cpp.src", ".cc"]

def add_error(errors, line, type, meta):
    if (line in errors):
        errors[line].append([type, meta])
    else:
        errors[line] = [type, meta]
    return errors

def loop(file_data, source=True):
    line = 0 
    fcn_data = {}

    errors = {}

    allow_class = ALLOW_CLASS_DEFS in file_data
    allow_header_impl = ALLOW_HEADER_IMPL in file_data
    allow_no_source = ALLOW_NO_SOURCE in file_data

    for l in file_data.split("\n"):
        if ("//" in l and "NOLINT" not in l and "namespace" not in l):
            l_ = "//" + l.split("//")[1]
            if (not re.match(ACCEPT_COMMENT_REGEX, l_)):
                errors = add_error(errors, line, Error.COMMENT, [line, line])

        if (re.match(FUNCTION_REGEX_SOURCE, l) and "for " not in l):
            header = l[:-2]
            meta = identify_function(file_data, header, line)
            meta.append(l)
            if (meta[2] > MAX_FCN):
                errors = add_error(errors, line, Error.FUNCTION_TOO_LONG, [header, meta])
            fcn_data[header] = meta
            if (not allow_no_source):   
                if (not source and not allow_header_impl):
                    errors = add_error(errors, line, Error.FUNCTION_IN_HEADER, [header, meta])


        if (re.match(CLASS_FIND_REGEX, l) and (not allow_class and source)):
            if ("default" not in l):
                errors = add_error(errors, line, Error.CLASS_IN_SOURCE, [line])
        line += 1

    return errors, fcn_data

def identify_function(file_data, header, line):
   
    start = file_data.find(header)
    start_l = line
    open = 0
    end = start
    stats = []
    lc = 0
    for li in file_data[start:].split("\n"):
        count_open = li.count("{")
        count_closed = li.count("}")
        open += count_open - count_closed
        end += len(li) + 1
        line += 1 
        lc += 1 if (li != "") else 0
        if open == 0:
            break 

    end_l = line
    stats = [start_l, end_l, lc]

    return stats

def identify_class_def_in_source(file_data):
    line = 0
    class_defs = []
    for l in file_data.split("\n"):
        if (re.match(CLASS_FIND_REGEX, l)):
            class_defs.append(line)
    return class_defs


def check_if_exists(files):
    for f in files:
        if (os.path.exists(f)):
            return True 


    return False


# def order_verify(source_file, header_file):


def launch_custom(file):
    with open(file) as f:
        file_data = f.read() 


    file_ending = "." + file.split(".")[-1]

    file_base = file.split(".")[0]
    check_files = []
    if (file_ending in header_endings):
        source = False

        check_files = [file_base + c for c in source_endings]
        if (not check_if_exists(check_files) and ALLOW_NO_SOURCE not in file_data.split("\n")[0]):
            print (ErrorText.SOURCE_FILE_MISSING.format(file=file))
    else:
        source = True

        check_files = [file_base + c for c in header_endings]
        if (not check_if_exists(check_files) and ALLOW_NO_HEADER not in file_data.split("\n")[0]):
            print (ErrorText.HEADER_FILE_MISSING.format(file=file))

    errors, fc = loop(file_data, source)

    # if (source == False):
    #     print (fc)

    for e in sorted(list(errors.keys())):
        meta = errors[e]
        error_code = meta[0]
        meta = meta[1]
        if (error_code == Error.COMMENT):
            print (ErrorText.CODE_WITH_COMMENT.format(file=file, line=meta[0]+1))
        elif (error_code == Error.FUNCTION_TOO_LONG):
            # print (meta)
            line_data = meta[1]
            start = line_data[0]
            end = line_data[1]
            code = line_data[-1]
            count = line_data[2]
            print (ErrorText.FUNCTION_TOO_LONG.format(file=file, line_count=count, line=start+1, code=code))
        elif (error_code == Error.FUNCTION_IN_HEADER):
            # print (meta)
            line_data = meta[1]
            start = line_data[0]
            end = line_data[1]
            code = line_data[-1]
            count = line_data[2]
            print (ErrorText.FUNCTION_IN_HEADER.format(file=file, line_count=count, line=start+1, code=code))

        elif (error_code == Error.CLASS_IN_SOURCE):
            print (ErrorText.CLASS_DEF_IN_SOURCE.format(file=file, line=meta[0]))

    if (errors == {}):
        return False 
    return True

