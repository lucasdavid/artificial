
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'artificial',
    'description': 'Implementation for Artificial Intelligence: '
                   'An Modern Approach book examples.',
    'author': 'Lucas David',
    'url': '',
    'download_url': '',
    'author_email': 'lucasolivdavid@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['artificial'],
    'scripts': [],
}

setup(**config)

