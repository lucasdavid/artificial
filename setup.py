from setuptools import setup, find_packages

setup(
    name='artificial',
    description='Implementations of Artificial Intelligence: '
                'An Modern Approach book examples.',
    long_description=open('README.md').read(),
    version='1.0',
    packages=find_packages(),
    scripts=[],
    author='Lucas David',
    author_email='lucasolivdavid@gmail.com',

    url='https://github.com/lucasdavid/artificial',
    download_url='https://github.com/lucasdavid/artificial/archive/master.zip',
    install_requires=['six', 'numpy', 'scipy'],
    tests_require=open('docs/requirements-dev.txt').readlines(),
)
