from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "bigram",
    ext_modules = cythonize("test.pyx"),
    )
