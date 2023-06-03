import setuptools
from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

cxx_flags = []
define_macros=[]

ext_modules =[]
cmdclass = {}
cmdclass["build_ext"] = cpp_extension.BuildExtension 

setup(
    name = "mole",
    packages = setuptools.find_packages(),
    version = "0.1.0",
    author="mole",
    author_email="mole",
    description="mole",
    
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)



