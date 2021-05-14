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
    scripts=['scripts/kinematic_replay/run_kinematic_replay',
        'scripts/kinematic_replay/run_kinematic_replay_ground',
        'scripts/neuromuscular_optimization/run_multiobj_optimization',
        'scripts/neuromuscular_optimization/run_neuromuscular_control',
        'scripts/sensitivity_analysis/run_sensitivity_analysis',
        'scripts/sensitivity_analysis/run_grid_search'
    ],
)
