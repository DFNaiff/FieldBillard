# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name="fieldbillard",
    version="0.0.1",
    description="Field Billard",
    packages=find_packages(where='fieldbillard'),
    author="Danilo de Freitas Naiff",
    author_email="dfnaiff@gmail.com",
    url="https://github.com/DFNaiff/FieldBillard/"
)