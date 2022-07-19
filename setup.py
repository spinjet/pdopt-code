# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

with open('README') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pdopt',
    version='0.1.0',
    description='Package for P-DOPT design exploration framework',
    long_description=readme,
    author='Andrea Spinelli',
    author_email='andrea.spinelli@cranfield.ac.uk',
    url='https://github.com/spin_jet/pdopt-code',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)