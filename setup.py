from setuptools import setup, find_packages

setup(
    name='jfti',
    version='0.5.0',
    description='library for reading and editing XMP tags',
    url='https://github.com/nycz/jfti',
    author='nycz',
    packages=find_packages(exclude=['testdata'])
)
