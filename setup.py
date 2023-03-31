from setuptools import setup

install_requires = [
    "numpy",
    "torch",
    "torchvision",
    "albumentations==0.4.6",
    "requests",
    "scikit-learn==1.2.2",
    "Pillow"
]

setup(
    name='segmentation',
    packages=['segmentation'],
    install_requires=install_requires,
    version='0.1.1',
    description='Duckietown segmentation library',
    include_package_data=True,
    author='Valentina-Gol',
)
