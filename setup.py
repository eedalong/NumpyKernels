import os
import sys
import re
import codecs
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools import dist
from setuptools.command.build_ext import build_ext as _build_ext
try:
    from Cython.Build import cythonize
except ImportError:

    def cythonize(*args, **kwargs):
        """cythonize"""
        from Cython.Build import cythonize
        return cythonize(*args, **kwargs)


class CustomBuildExt(_build_ext):
    """CustomBuildExt"""

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

compile_extra_args = ["-std=c++11", "-O3", "-fopenmp"]
link_extra_args = ["-fopenmp"]

if sys.platform == "darwin":
    compile_extra_args = ['-std=c++11', "-mmacosx-version-min=10.9"]
    link_extra_args = ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

extensions = [
    Extension(
        "npkernel.kernels",
        ["kernels/array_slice.pyx"],
        language="c++",
        extra_compile_args=compile_extra_args,
        extra_link_args=link_extra_args, ),
]
def get_package_data(path):
    files = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return

setup(
    name="npkernel",
    description='Cython Kernels For Numpy Operations',
    version="1.0.0",
    long_description="NULL",
    long_description_content_type='text/markdown',
    url="https://github.com/eedalong/NumpyKernels",
    setup_requires=[
        'setuptools>=18.0',
        'numpy>=1.16.4',
    ],
    install_requires=[],
    cmdclass={'build_ext': CustomBuildExt},
    packages=find_packages(),
    include_package_data=False,
    ext_modules=cythonize(extensions),
    #ext_modules=extensions,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
