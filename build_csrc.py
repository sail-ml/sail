import subprocess
subprocess.run(["doxygen"], cwd="docs")
subprocess.run(["sphinx-build", "-b", "html", ".", "./sphinx"], cwd="docs")
subprocess.run(["python", "setup.py", "develop"], cwd="sail/csrc")