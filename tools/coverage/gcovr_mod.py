import gcovr, sys, os, re
import gcovr.__main__ as gc_main
import gcovr.configuration as config
import gcovr.utils as utils
from argparse import ArgumentParser
from os.path import normpath
import random

whitelist = ["if (", "else ", "for (", "switch ("]
regex_whitelist = ["\s*case .*::"]

def check_branch(code):
    for r in whitelist:
        if (r in code):
            return True
    for r in regex_whitelist:
        if (re.match(r, code)):
            return True
    return False

def if_branch_reduce(file, line, line_code):
    if ("dtype" not in file):
        if ("&&" in line_code or "||" in line_code):
            if "if (ndim > 1 || i % kMaxItemNumPerLine == 0) {" not in line_code:
                return line 

    branches = [False, False] # exec, no-exec
    call_count = 0
    no_call_count = 0
    call_branch = None 
    no_call_branch = None
    for b in line.branches:
        branch = line.branches[b]
        if (not call_branch):
            call_branch = branch 
        if (not no_call_branch):
            no_call_branch = branch

        if (branch.fallthrough):
            call_count += branch.count
        else:
            no_call_count += branch.count 
        if (branch.is_covered):
            if (branch.fallthrough):
                branches[0] = True
                call_branch = branch 
            else:
                branches[1] = True 
                no_call_branch = branch
        

    line.branches = {1: call_branch, 2: no_call_branch}
    return line
        

def branch_reduce(file, line, line_code): 

    if (len(line.branches) <= 2):
        return line

    if ("if (" in line_code):
        return if_branch_reduce(file, line, line_code)

    if ("for (" not in line_code):
        return line


    branches = [False, False] # exec, no-exec
    call_count = 0
    no_call_count = 0
    call_branch = None 
    no_call_branch = None
    for b in line.branches:
        branch = line.branches[b]
        if (branch == None):
            continue
        if (branch.fallthrough):
            call_count += branch.count
        else:
            no_call_count += branch.count 
        if (branch.is_covered):
            if (branch.fallthrough):
                branches[0] = True
                call_branch = branch 
            else:
                branches[1] = True 
                no_call_branch = branch
        else:
            if (not call_branch):
                call_branch = branch 
            if (not no_call_branch):
                no_call_branch = branch

    line.branches = {1: call_branch, 2: no_call_branch}
    return line

def fail_under(covdata, threshold_line, threshold_branch):
    (lines_total, lines_covered, percent,
        branches_total, branches_covered,
        percent_branches) = utils.get_global_stats(covdata)

    if branches_total == 0:
        percent_branches = 100.0

    if percent < threshold_line and percent_branches < threshold_branch:
        sys.exit(6)
    if percent < threshold_line:
        sys.exit(2)
    if percent_branches < threshold_branch:
        sys.exit(4)
        
def main(args=None):
    parser = gc_main.create_argument_parser()
    
    cli_options = parser.parse_args(args=args)

    cfg_name = gc_main.find_config_name(cli_options)
    cfg_options = {}
    if cfg_name is not None:
        with io.open(cfg_name, encoding='UTF-8') as cfg_file:
            cfg_options = gc_main.parse_config_into_dict(
                gc_main.parse_config_file(cfg_file, filename=cfg_name))

    options_dict = gc_main.merge_options_and_set_defaults(
        [cfg_options, cli_options.__dict__])
    options_dict["html_tab_size"] = 4
    options = gc_main.Options(**options_dict)

    logger = gc_main.Logger(options.verbose)


    if cli_options.version:
        logger.msg(
            "gcovr {version}\n"
            "\n"
            "{copyright}",
            version=gc_main.__version__, copyright=gc_main.COPYRIGHT)
        sys.exit(0)

    if options.html_title == '':
        logger.error(
            "an empty --html_title= is not allowed.")
        sys.exit(1)

    if options.html_medium_threshold == 0:
        logger.error(
            "value of --html-medium-threshold= should not be zero.")
        sys.exit(1)

    if options.html_medium_threshold > options.html_high_threshold:
        logger.error(
            "value of --html-medium-threshold={} should be\n"
            "lower than or equal to the value of --html-high-threshold={}.",
            options.html_medium_threshold, options.html_high_threshold)
        sys.exit(1)

    if options.html_tab_size < 1:
        logger.error(
            "value of --html-tab-size= should be greater 0.")
        sys.exit(1)


    potential_html_output = (
        (options.html and options.html.value)
        or (options.html_details and options.html_details.value)
        or (options.output))
    if options.html_details and not potential_html_output:
        logger.error(
            "a named output must be given, if the option --html-details\n"
            "is used.")
        sys.exit(1)

    if options.objdir is not None:
        if not options.objdir:
            logger.error(
                "empty --object-directory option.\n"
                "\tThis option specifies the path to the object file "
                "directory of your project.\n"
                "\tThis option cannot be an empty string.")
            sys.exit(1)
        tmp = options.objdir.replace('/', os.sep).replace('\\', os.sep)
        while os.sep + os.sep in tmp:
            tmp = tmp.replace(os.sep + os.sep, os.sep)
        if normpath(options.objdir) != tmp:
            logger.warn(
                "relative referencing in --object-directory.\n"
                "\tthis could cause strange errors when gcovr attempts to\n"
                "\tidentify the original gcc working directory.")
        if not os.path.exists(normpath(options.objdir)):
            logger.error(
                "Bad --object-directory option.\n"
                "\tThe specified directory does not exist.")
            sys.exit(1)

    options.starting_dir = os.path.abspath(os.getcwd())
    if not options.root:
        logger.error(
            "empty --root option.\n"
            "\tRoot specifies the path to the root "
            "directory of your project.\n"
            "\tThis option cannot be an empty string.")
        sys.exit(1)
    options.root_dir = os.path.abspath(options.root)

    #
    # Setup filters
    #

    # The root filter isn't technically a filter,
    # but is used to turn absolute paths into relative paths
    options.root_filter = re.compile('^' + re.escape(options.root_dir + os.sep))

    if options.exclude_dirs is not None:
        options.exclude_dirs = [
            f.build_filter(logger) for f in options.exclude_dirs]

    options.exclude = [f.build_filter(logger) for f in options.exclude]
    options.filter = [f.build_filter(logger) for f in options.filter]
    if not options.filter:
        options.filter = [gc_main.DirectoryPrefixFilter(options.root_dir)]

    options.gcov_exclude = [
        f.build_filter(logger) for f in options.gcov_exclude]
    options.gcov_filter = [f.build_filter(logger) for f in options.gcov_filter]
    if not options.gcov_filter:
        options.gcov_filter = [gc_main.AlwaysMatchFilter()]

    # Output the filters for debugging
    for name, filters in [
        ('--root', [options.root_filter]),
        ('--filter', options.filter),
        ('--exclude', options.exclude),
        ('--gcov-filter', options.gcov_filter),
        ('--gcov-exclude', options.gcov_exclude),
        ('--exclude-directories', options.exclude_dirs),
    ]:
        logger.verbose_msg('Filters for {}: ({})', name, len(filters))
        for f in filters:
            logger.verbose_msg('- {}', f)

    covdata = dict()
    if options.add_tracefile:
        gc_main.collect_coverage_from_tracefiles(covdata, options, logger)
    else:
        gc_main.collect_coverage_from_gcov(covdata, options, logger)

    covdata_keys = list(covdata.keys())
    random.shuffle(covdata_keys)
    covdata = {c: covdata[c] for c in covdata_keys if ("xsimd" not in c and "test/" not in c)}
    for i in covdata:
        file = i
        lines = list(covdata[i].lines.keys())
        with open(i, "r") as f:
            line_code = f.read().split("\n")

        for line in lines:
            line = covdata[i].lines[line]
            line_idx = line.lineno
            # if ("DECLARE_DISPATCH" in line_code[line_idx]):
            #     print ("ja")
            #     line.count = 1
            #     print (line.is_covered)
            if line.branches != {}:
                
                keep = False
                code = line_code[line_idx-1]
                for b in line.branches:
                    
                    if check_branch(code):
                        keep = True
                        break
                    else:
                        continue
                if (not keep):
                    line.branches = {}
                else:
                    line = branch_reduce(file, line, code)
                    covdata[i].lines[line_idx] = line
    error_occurred = gc_main.print_reports(covdata, options, logger)
    if error_occurred:
        logger.error(
            "Error occurred while printing reports"
        )
        sys.exit(7)


    # if options.fail_under_line > 0.0 or options.fail_under_branch > 0.0:
    fail_under(covdata, 90, 90)

if __name__ == "__main__":
    main()