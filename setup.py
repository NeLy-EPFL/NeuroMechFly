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
    Extension("NeuroMechFly.network.network_generator",
              ["NeuroMechFly/network/network_generator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("NeuroMechFly.network.oscillator",
              ["NeuroMechFly/network/oscillator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("NeuroMechFly.network.neuron",
              ["NeuroMechFly/network/neuron.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    "NeuroMechFly/container/*.pyx"
]

setuptools.setup(
    name='NeuroMechFly',
    version='0.1',
    description='Modules to run NeuroMechFly simulation',
    #url='https://gitlab.com/FARMSIM/farms_network.git',
    author='Neuroengineering Lab.',
    author_email='victor.lobatorios@epfl.ch',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'farms_pylog',
        'numpy',
        'pandas',
        'matplotlib==3.0.2',
        'networkx==2.3',
        'pydot',
        'ddt',
        'scipy==1.5.4',
        'trimesh',
        'treelib',
        'tqdm',
        'pybullet',
        'PyYAML',
        'ipython',
        'dataclasses',
        'jmetalpy',
        'tables',
        'pillow'
    ],
    dependency_links=[
        'https://gitlab.com/FARMSIM/farms_pylog.git',
    ],
    zip_safe=False,
    ext_modules=cythonize(extensions),
    package_data={
        'network': ['*.pxd'],
        'container': ['*.pxd'],
    },
)
