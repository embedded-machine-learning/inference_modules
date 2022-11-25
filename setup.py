import io, os, re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name='hw_modules',
    version="0.1.0",

    url='https://github.com/embedded-machine-learning/inference_modules/tree/master',
    author='CDL EML',
    license='Apache 2.0',
    author_email="cdleml@tuwien.ac.at",

    description="Inference Modules for DNN execution",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[],

    classifiers=[
        'Programming Language :: Python :: 3.6'
    ],
)
