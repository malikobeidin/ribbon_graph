long_description =  """\
This is a package for manipulating ribbon graphs
"""

import re, sys, subprocess, os, shutil, glob, sysconfig
from setuptools import setup, Command
from setuptools.command.build_py import build_py





# Get version number from module
version = re.search("__version__ = '(.*)'",
                    open('__init__.py').read()).group(1)

setup(
    name = 'ribbon_graph',
    version = version,
    description = 'Ribbon Graphs',
    long_description = long_description,
    url = 'https://bitbucket.org/mobeidin/ribbon_graph',
    author = 'Malik Obeidin',
    author_email = 'mobeidin@illiois.edu',
    license='GPLv2+',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
        ],

    packages = ['ribbon_graph'],
    package_dir = {'ribbon_graph':''},
    ext_modules = [],
    zip_safe = False,
)

