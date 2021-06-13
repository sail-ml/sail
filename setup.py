# import subprocess
# subprocess.run(["python", "setup.py", "develop"], cwd="sail/csrc")

import os
import re
import sys
import sysconfig
import subprocess
import platform
import setuptools
import cpufeature
import subprocess
import glob, pathlib
from shutil import copyfile
import numpy as np 

import distutils
from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc
import distutils.sysconfig as sysconfig

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

build_path = ""

allow_avx = True
coverage = False

class CICommand(distutils.cmd.Command):

    description = 'build for ci (as in no avx)'
    user_options = [
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


    def run(self):
        global allow_avx
        global coverage
        allow_avx = False
        coverage = True


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
        subprocess.run(["rm", "-rf", "build/*"])
        subprocess.run(["python", "process_docs.py"], cwd="sail")
        global allow_avx
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)
        global build_path
        build_path = extdir.parent.absolute()


        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config,
            '-DPYTHON_INCLUDE_DIR=' + get_python_inc(),
            '-DPYTHON_LIBRARY=' + sysconfig.get_config_var('LIBDIR'),
            '-DPYTHON_EXECUTABLE=' + sys.executable + "P",
            '-DPYTHON_NUMPY_INCLUDE_DIR=' + np.get_include()
            # '-DCMAKE_C_COMPILER=/usr/bin/gcc',
            # '-DCMAKE_CXX_COMPILER=/usr/bin/g++-8'
        ]

        print (cmake_args)

        print (cpufeature.CPUFeature)
        if (cpufeature.CPUFeature["AVX2"] and allow_avx):
            print ("Compiling Sail with AVX2 Support")
            cmake_args.append("-DUSE_AVX2=ON")
            cmake_args.append("-DUSE_MKL=ON")
        else:
            print ("Compiling Sail without AVX2 Support")
            cmake_args.append("-DUSE_AVX2=OFF")
            cmake_args.append("-DUSE_MKL=OFF")

        if (coverage):
            cmake_args.append("-DCOVERAGE=ON")

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        print ("executing build")
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible 
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))

        print (build_path)

        copyfile("%s/libsail_c.so" % build_path, "sail/csrc/libsail_c.so")
        copyfile("%s/libmodules.so" % build_path, "sail/modules/libmodules.so")
        copyfile("%s/liblosses.so" % build_path, "sail/losses/liblosses.so")
        copyfile("%s/liboptimizers.so" % build_path, "sail/optimizers/liboptimizers.so")

        copyfile("%s/libmodules.so" % build_path, "%s/../modules/libmodules.so" % build_path)
        copyfile("%s/liblosses.so" % build_path, "%s/../losses/liblosses.so" % build_path)
        copyfile("%s/liboptimizers.so" % build_path, "%s/../optimizers/liboptimizers.so" % build_path)

def s():
   
    save_gen = False
    if "save-gen" in sys.argv:
        save_gen = True
        sys.argv.remove("save-gen")

    files = glob.glob("sail/csrc/core/**/*.cpp*", recursive=True)
    files = list(files) + list(glob.glob("sail/csrc/core/**/*.h*", recursive=True))
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


    setup(
        name='sail-ml',
        version='0.0.1a1',
        author='Tucker Siegel',
        author_email='tgsiegel@umd.edu',
        description='SAIL: Simple AI Library',
        long_description='SAIL is a python package designed for speed and simplicity when developing and running deep learning models. Built on top of a c++ library with python bindings, SAIL is currently in development, changes are being released daily with new features and bug fixes.',
        url="https://sailml.org",
        keywords='sail sail-ml machine learning',
        packages = [
            "sail", 
            "sail.csrc",
            "sail.modules",
            "sail.losses",
            "sail.optimizers",
            ],#setuptools.find_packages(),
        ext_modules=[CMakeExtension('sail.csrc.libsail_c')],
        cmdclass={'build_ext': CMakeBuild, "ci": CICommand},
        install_requires=REQUIREMENTS
        # cmdclass=dict(build_ext=CMakeBuild),
    )

    if (not save_gen):
        for f in created_names:
            os.remove(f)
            print (f)

s()

