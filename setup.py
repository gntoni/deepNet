from setuptools import setup
from setuptools import find_packages

install_requires = [
        'numpy',
        'xmltodict',
        'theano',
        'lasagne',
        ]


setup(
      name="deepNet",
      version="0.1.0",
      description="Base class to create Deep Networks with Lasagne",
      author="Toni Gabas",
      author_email="a.gabas@aist.go.jp",
      url="",
      long_description=
      """
        Base class with functions to create, train and test any
        neural network model with Theano+Lasagne.
      """,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      )
