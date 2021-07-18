import argparse
from linter import execute

def parse_args():

    parser = argparse.ArgumentParser(description="clang-tidy wrapper")
    
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to a clang-tidy config file. Defaults to '.clang-tidy'.",
    )
    parser.add_argument(
        "--base-path",
        default="sail/csrc/core",
        help="Base path to search for files to use",
    )
    parser.add_argument(
        "--include",
        help="Single file to use",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
   
    return parser.parse_args()


def run():
    options = parse_args()
    execute(options)

if __name__ == "__main__":
    run()