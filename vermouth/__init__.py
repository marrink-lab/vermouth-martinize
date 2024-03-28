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
    from importlib.resources import files, as_file
    import atexit
    from contextlib import ExitStack
except ImportError:
    from pathlib import Path
    DATA_PATH = Path(__file__).parent / 'data'
    del Path
else:
    ref = files('vermouth') / 'data'
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    DATA_PATH = file_manager.enter_context(as_file(ref))
    del files, as_file, atexit, ExitStack

del pbr

from scipy.spatial import cKDTree as KDTree

del LOGGER

from .molecule import Molecule  # pylint: disable=wrong-import-position
from .processors import *  # pylint: disable=wrong-import-position
from .system import System  # pylint: disable=wrong-import-position
