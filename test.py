import subprocess
import argparse
import glob, os, sail
import shutil

from tools.coverage import gcovr_mod
# gcovr_mod.main(None)
# exit()

BASE_COVERAGE_COMMAND = ["python", "tools/coverage/gcovr_mod.py", "--exclude-unreachable-branches", "--exclude-throw-branches", "-e", '\".*xsimd.*\"', "--filter", "sail/", "-s"]

def run_cpp_tests():
    folder = glob.glob("build/t*/sail/csrc")[0]
    subprocess.run(["ctest"], cwd=folder)

def run_python_tests():
    subprocess.run(["python", "python_test/run.py"])

def generate_html_coverage():
    command = BASE_COVERAGE_COMMAND + ["--html", "--html-details", "-o", "coverage/coverage.html"]
    if (not os.path.exists("coverage/")):
        os.mkdir("coverage")
    else:
        shutil.rmtree('coverage')
        os.mkdir("coverage")

    result = subprocess.run(command, stderr=subprocess.DEVNULL)
    if (result.returncode != 0):
        raise Exception("Coverage not high enough")

def generate_xml_coverage():
    command = BASE_COVERAGE_COMMAND + ["--xml", "coverage.xml"]
    result = subprocess.run(command, stderr=subprocess.DEVNULL)
    if (result.returncode != 0):
        raise Exception("Coverage not high enough")

def generate_coverage():
    command = BASE_COVERAGE_COMMAND
    result = subprocess.run(command, stderr=subprocess.DEVNULL)
    if (result.returncode != 0):
        raise Exception("Coverage not high enough")

def dispatch(parser):

    options = parser.parse_args()

    python = options.python 
    c = options.cpp
    all_ = options.all
    coverage = options.coverage
    html = options.html
    xml = options.xml

    if (all_ or c):
        run_cpp_tests()

    if (all_ or python):
        run_python_tests()

    if (html):
        generate_html_coverage()
    elif (xml):
        generate_xml_coverage()
    elif (coverage):
        generate_coverage()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="testing wrapper")
    
    parser.add_argument(
        "--python",
        "-p",
        action="store_true",
        help="Run python tests",
    )
    parser.add_argument(
        "--cpp",
        "-c",
        action="store_true",
        help="Run c++ tests",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run c++ and python tests"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage and display in terminal"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate coverage in html file"
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate coverage in xml file"
    )

    dispatch(parser)
