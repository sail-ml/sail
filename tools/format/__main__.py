import subprocess, glob



if __name__ == "__main__":

    files = list(glob.glob("sail/**/*.cpp*", recursive=True))
    subprocess.run(["clang-format", "-i"] + files)