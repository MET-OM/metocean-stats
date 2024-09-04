#!/usr/bin/env python

import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'version.py')).read())

setuptools.setup(
    name        = 'metocean-stats',
    description = 'metocean-stats - Tool for generation of metocean statistics',
    author      = 'Konstantinos Christakos MET Norway & NTNU',
    url         = 'https://github.com/MET-OM/metocean-stats',
    download_url = 'https://github.com/MET-OM/metocean-stats',
    version = __version__,
    license = 'LGPLv3',
    install_requires = [
        'numpy>=1.17',
        'xarray',
        'matplotlib>=3.1',
        'pandas',
        'pip',
        'scipy',
        'windrose',
        'seaborn>=0.12.2',
        'pyextremes',
        'cartopy',
        'python-docx',
    ],
    packages = setuptools.find_packages(),
    include_package_data = True,
    setup_requires = ['setuptools_scm'],
    tests_require = ['pytest'],
    scripts = []
)
