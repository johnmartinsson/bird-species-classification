try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Bird Species Classification using Convolutional Neural Networks',
    'author': 'John Martinsson',
    'url': 'https://github.com/johnmartinsson/bird-species-classification.git',
    'download_url': 'https://github.com/johnmartinsson/bird-species-classification/archive/master.zip',
    'author_email': 'john.martinsson@gmail.com',
    'version': '0.1',
    'install_requires': ['scipy', 'keras', 'nose'],
    'packages': ['bird'],
    'scripts': [],
    'name': 'bird'
}

setup(**config)
