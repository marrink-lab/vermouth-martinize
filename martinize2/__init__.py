# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:41:47 2017

@author: Peter Kroon
"""

# Find the data directory once.
import os
try:
    import pkg_resources
except ImportError:
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
else:
    DATA_PATH = pkg_resources.resource_filename('martinize2', 'data')
    del pkg_resources
del os

from .martinize2 import *
