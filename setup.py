from setuptools import setup, find_packages

setup(name='dropo',
      version='2.1',
      install_requires=['numpy','nevergrad'],  # And any other dependencies foo needs
      packages=find_packages(exclude=["datasets"])
)