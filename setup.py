from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "torch",
    "torchvision",
    "albumentations==0.4.6",
    "requests"
]

setup(
    name='segmentation',
    packages=find_packages(include=['segmentation']),
    install_requires = install_requires,
    version='0.1.0',
    description='Duckietown segmentation library',
    author='Valentina-Gol',
)
