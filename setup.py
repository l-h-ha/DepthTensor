from setuptools import setup, Extension, find_namespace_packages
import pybind11
import glob
import os

source_package = "depthtensor" if "depthtensor" in os.listdir("src") else "DepthTensor"
source_packages = find_namespace_packages(
    where="src",
    include=[source_package, f"{source_package}.*"],
)
packages = [package.replace(source_package, "depthtensor", 1) for package in source_packages]

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

setup(
    packages=packages,
    package_dir={"depthtensor": f"src/{source_package}"},
    package_data={"depthtensor._ext": ["*.pyi"]},
    ext_modules=ext_modules,
)
