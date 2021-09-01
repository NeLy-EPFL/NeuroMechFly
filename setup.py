import numpy
import setuptools

from setuptools import setup, dist, find_packages
from setuptools.extension import Extension

from farms_container import get_include

dist.Distribution().fetch_build_eggs(['numpy'])
import numpy

dist.Distribution().fetch_build_eggs(['Cython>=0.15.1'])
from Cython.Build import cythonize
from Cython.Compiler import Options

DEBUG = False

Options.docstrings = False
Options.fast_fail = True
Options.annotate = True
Options.warning_errors = True
Options.profile = False

extensions = [
    Extension("NeuroMechFly.simulation.bullet_utils",
              ["NeuroMechFly/simulation/bullet_utils.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
]

setuptools.setup(
    name='NeuroMechFly',
    version='0.1',
    description='Modules to run NeuroMechFly simulation',
    author='Neuroengineering Lab.',
    author_email='NeuroMechFly@groupes.epfl.ch',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'farms_pylog @ git+https://gitlab.com/FARMSIM/farms_pylog.git',
        'farms_network @ git+https://gitlab.com/FARMSIM/farms_network.git',
        'farms_container @ git+https://gitlab.com/FARMSIM/farms_container.git',
        'df3dPostProcessing @ git+https://github.com/NeLy-EPFL/df3dPostProcessing.git',
        'numpy',
        'pandas',
        'matplotlib',
        'networkx',
        'scipy',
        'treelib',
        'trimesh',
        'tqdm',
        'pybullet',
        'PyYAML',
        'dataclasses',
        'jmetalpy',
        'tables',
        'pillow',
        'shapely',
        'scikit-posthocs'
    ],
    ext_modules=cythonize(
        extensions,
        include_path=[numpy.get_include()] + [get_include()],
        annotate=True,
        compiler_directives={
            'embedsignature': True,
            'cdivision': True,
            'language_level': 3,
            'infer_types': True,
            'profile': True,
            'wraparound': False,
            'boundscheck': DEBUG,
            'nonecheck': DEBUG,
            'initializedcheck': DEBUG,
            'overflowcheck': DEBUG,
        }
    ),
    scripts=['scripts/kinematic_replay/run_kinematic_replay',
        'scripts/kinematic_replay/run_kinematic_replay_ground',
        'scripts/neuromuscular_optimization/run_multiobj_optimization',
        'scripts/neuromuscular_optimization/run_neuromuscular_control',
        'scripts/sensitivity_analysis/run_sensitivity_analysis',
        'scripts/sensitivity_analysis/run_grid_search'
    ],
    package_data={
        'NeuroMechFly': ['*.pxd'],
        'farms_container': ['*.pxd'],
    },
)
