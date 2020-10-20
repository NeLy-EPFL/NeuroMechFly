import setuptools
from Cython.Build import cythonize
from Cython.Compiler import Options
from distutils.extension import Extension
import numpy

Options.docstrings = True
Options.fast_fail = True
Options.annotate = True
Options.warning_errors = True

# directive_defaults = Cython.Compiler.Options.get_directive_defaults()

extensions = [
    Extension("network.network_generator",
              ["network/network_generator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("network.oscillator",
              ["network/oscillator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    "container/*.pyx"
]

setuptools.setup(
    name='NeuroMechFly',
    version='0.1',
    description='Modules to run NeuroMEchFly simulation',
    #url='https://gitlab.com/FARMSIM/farms_network.git',
    author='Neuroengineering Lab.',
    author_email='victor.lobatorios@epfl.ch',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib==3.0.2',
        'networkx',
        'pydot',
        'ddt',
        'scipy',
        'trimesh',
        'treelib',
        'tqdm',
        'pybullet',
        'PyYAML',
        'ipython',
        'dataclasses',
        'jmetalpy',
        'tables'
    ],
    zip_safe=False,
    ext_modules=cythonize(extensions),
    package_data={
        'network': ['*.pxd'],
        'container': ['*.pxd'],
    },
)
