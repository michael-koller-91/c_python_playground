from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy


ext = Extension(name='lablas', sources=['lablas.pyx'])
setup(
    ext_modules=cythonize(ext, annotate=True),
    include_dirs=[numpy.get_include()]
)
