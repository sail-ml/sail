# import subprocess
# subprocess.run(["python", "setup.py", "develop"], cwd="sail/csrc")

import os
import re
import sys
import sysconfig
import platform
import subprocess
import glob

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # print (dir(ext))
        # print (ext.sourcedir)
        # print (self.get_ext_fullpath(ext.name))

        extdir = os.path.abspath(ext.sourcedir)# + "/sail/csrc")
        # extdir = os.path.abspath(
        #     os.path.dirname(self.get_ext_fullpath(ext.name)))
        print (extdir)
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath("sail/csrc"),
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # if platform.system() == "Windows":
        #     cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
        #         cfg.upper(),
        #         extdir)]
        #     if sys.maxsize > 2**32:
        #         cmake_args += ['-A', 'x64']
        #     build_args += ['--', '/m']
        # else:
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print (cmake_args)
        print (build_args)
        # print (ext.sourcedir)
        # print (self.build_temp)
        # exit()
        # exit()

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)

        # subprocess.check_call(["mv", "libsail_c.so", "sail/csrc/libsail_c.so"])

        print()  # Add an empty line for cleaner output
        
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

setup(
    name='sail',
    version='0.0.1a',
    # author='Benjamin Jack',
    # author_email='benjamin.r.jack@gmail.com',
    # description='A hybrid Python/C++ test project',
    # long_description='',
    # add extension module
    packages=["sail", "sail.csrc"],
    package_data={"sail": [os.path.abspath("sail/csrc/libsail_c.so")]},
    ext_modules=[CMakeExtension('sail_c')],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    # zip_safe=False,
)

for f in created_names:
    # os.remove(f)
    print (f)