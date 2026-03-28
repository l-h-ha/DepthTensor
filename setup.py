from setuptools import setup, Extension
import pybind11
import glob
import os

bindings_src = glob.glob("csrc/bindings/*.cpp")
cpu_core_src = glob.glob("csrc/cpu/core/*.cpp")
cpu_nn_src = glob.glob("csrc/cpu/nn/*.cpp")

all_sources = bindings_src + cpu_core_src + cpu_nn_src

ext_modules = [
    Extension(
        "depthtensor._ext",
        sources=all_sources,
        include_dirs=[
            pybind11.get_include(),
            os.path.abspath("csrc/include"),
        ],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-shared", "-fPIC"],
    ),
]

setup(ext_modules=ext_modules)
