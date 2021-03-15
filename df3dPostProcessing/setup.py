import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="df3dPostProcessing",
    version="1.0",
    packages=["df3dPostProcessing"],
    author="Victor Lobato",
    author_email="victor.lobatorios@epfl.ch",
    description="Postprocessing functions for DeepFly3D results",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/df3dPostProcessing",
    install_requires=["numpy", "scipy", "matplotlib","opencv-python","opencv-contrib-python","seaborn","scikit-learn","pandas","ikpy"],
)
