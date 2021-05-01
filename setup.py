import numpy
import setuptools

setuptools.setup(
    name='NeuroMechFly',
    version='0.1',
    description='Modules to run NeuroMechFly simulation',
    author='Neuroengineering Lab.',
    author_email='victor.lobatorios@epfl.ch',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'farms_pylog @ git+https://gitlab.com/FARMSIM/farms_pylog.git',
        'farms_network @ git+https://gitlab.com/FARMSIM/farms_network.git',
        'farms_container @ git+https://gitlab.com/FARMSIM/farms_container.git',
        'df3dPostProcessing @ git+https://github.com/NeLy-EPFL/df3dPostProcessing.git',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'treelib',
        'tqdm',
        'pybullet',
        'PyYAML',
        'dataclasses',
        'jmetalpy',
        'tables',
        'pillow'
    ],
    scripts=['scripts/run_kinematic_replay',
        'scripts/run_kinematic_replay_ground'
        'scripts/run_multiobj_optimization',
        'scripts/run_neuromuscular_control',
        'scripts/run_sensitivity_analysis'
    ],
)
