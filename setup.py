# import subprocess
# subprocess.run(["python", "setup.py", "develop"], cwd="sail/csrc")

import os
import re
import sys
import sysconfig
import platform
import setuptools
import subprocess
import glob, pathlib
from shutil import copyfile


from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible 
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))
        
files = glob.glob("sail/csrc/src/**/*.cpp*", recursive=True)
files = list(files) + list(glob.glob("sail/csrc/src/**/*.h*", recursive=True))
files = list(files) + list(glob.glob("sail/csrc/python/**/*.cpp*", recursive=True))
files = list(files) + list(glob.glob("sail/csrc/python/**/*.h*", recursive=True))

os.system("clang-format -i " + " ".join(files))
src_files = glob.glob("**/*.src", recursive=True)
print (src_files)
os.system("python template_converter.py " + " ".join(src_files))
created_names = []
for n in src_files:
    (base, ext) = os.path.splitext(n)
    newname = base
    created_names.append(newname)

# print (setuptools.find_packages(
#         where = './',
#     ))

print (setuptools.find_packages())
print (setuptools.find_packages())
print (setuptools.find_packages())
print (setuptools.find_packages())
print (setuptools.find_packages())

# exit()

setup(
    name='sail',
    version='0.0.1a',
    # author='Benjamin Jack',
    # author_email='benjamin.r.jack@gmail.com',
    # description='A hybrid Python/C++ test project',
    # long_description='',
    # add extension module
    packages = ["sail", "sail.csrc"],#setuptools.find_packages(),
    # ext_modules=[CMakeExtension('DolphinTrading/indicator_bindings')],
    # packages=["sail", "sail.csrc"],
    # package_data={"sail": [os.path.abspath("sail/csrc/libsail_c.so")]},
    ext_modules=[CMakeExtension('sail.csrc.libsail_c')],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    # zip_safe=False,
)

for f in created_names:
    os.remove(f)
    print (f)


copyfile("build/lib.linux-x86_64-3.7/sail/csrc/libsail_c.so", "sail/csrc/libsail_c.so")