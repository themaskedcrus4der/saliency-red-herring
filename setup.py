#!/usr/bin/env python
import os
from setuptools import setup

setup(
    name='activmask',
    description='A way to get your CNN to ignore confounding features.',
    version='0.0.1',
    author='A very mysterious person.',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'activmask = activmask.main:main'
        ]
    }

)
