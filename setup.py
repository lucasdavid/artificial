try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='artificial',
    description='Implementation for Artificial Intelligence: '
                'An Modern Approach book examples.',
    long_description=open('README.md').read(),
    version='0.1',
    packages=['artificial'],
    scripts=[],
    author='Lucas David',
    author_email='lucasolivdavid@gmail.com',

    url='https://github.com/lucasdavid/artificial',
    download_url='https://github.com/lucasdavid/artificial/archive/master.zip',
    install_requires=['numpy', 'scipy'],
    tests_require=open('requirements-dev.txt').readlines(),
)
