from setuptools import setup, find_packages

setup(
    name='tomaat',
    version="0.1.3",
    description='TOMAAT SERVER SIDE',
    url='https://github.com/fmilletari/tomaat',
    author='Fausto Milletari',
    author_email='tomaat.segmentation@gmail.com',
    packages=find_packages(),
    install_requires=[
        'klein',
        'SimpleITK',
        'numpy',
        'requests',
        'click',
        'tinydb',
    ],
)
