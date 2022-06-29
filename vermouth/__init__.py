# -*- coding: utf-8 -*-
# Copyright 2018 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
VerMoUTH: The Very Modular Universal Transformation Helper

Provides functionality for creating MD topologies from coordinate files. Powers
the CLI tool martinize2.
"""
import logging

import pbr.version

from .log_helpers import StyleAdapter, get_logger

__version__ = pbr.version.VersionInfo('vermouth').release_string()

logging.getLogger(__name__).addHandler(logging.NullHandler())

LOGGER = StyleAdapter(get_logger(__name__))


# Find the data directory once.
try:
    import pkg_resources
except ImportError:
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
    del os
else:
    DATA_PATH = pkg_resources.resource_filename('vermouth', 'data')
    del pkg_resources

del pbr

from scipy.spatial import cKDTree as KDTree

del LOGGER

from .molecule import Molecule  # pylint: disable=wrong-import-position
from .processors import *  # pylint: disable=wrong-import-position
from .system import System  # pylint: disable=wrong-import-position
