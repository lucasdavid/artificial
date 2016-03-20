
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'artificial',
    'description': 'Implementation for Artificial Intelligence: '
                   'An Modern Approach book examples.',
    'long_description': open('README.md').read(),
    'author': 'Lucas David',
    'author_email': 'lucasolivdavid@gmail.com',
    'url': 'https://github.com/lucasdavid/artificial',
    'download_url': 'https://github.com/lucasdavid/artificial/'
                    'archive/master.zip',
    'version': '0.1',
    'install_requires': [],
    'tests_require': ['nose', 'coverage'],
    'packages': ['artificial'],
    'scripts': [],
}

setup(**config)
