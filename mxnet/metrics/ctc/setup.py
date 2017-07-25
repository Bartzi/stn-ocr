from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

setup(
   ext_modules=cythonize([
       Extension(
           "ctc_loss",
           sources=["ctc_loss.pyx"],
           libraries=["warpctc"],
           include_dirs=[numpy.get_include()],
           language="c++",
       )
   ])
)