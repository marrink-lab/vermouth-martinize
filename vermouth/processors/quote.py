#!/usr/bin/env python3
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
Reads quotes, and produces a random one.
"""

import os.path
import random

from .processor import Processor
from .. import DATA_PATH
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))

QUOTE_FILE = os.path.join(DATA_PATH, 'quotes.txt')


def read_quote_file(filehandle):
    """
    Iterates over `filehandle`, and yields all strings that are not empty.

    Parameters
    ----------
    filehandle: collections.abc.Iterable[str]
        A file opened for reading.

    Yields
    ------
    str
        All stripped elements of `filehandle` that are not empty.
    """
    for line in filehandle:
        line = line.strip()
        if line:
            yield line


class Quoter(Processor):
    """
    Processor that can produce random string taken from a file. Useful for e.g.
    quotes.

    Parameters
    ----------
    quote_file: pathlib.Path or str
        The path of the file containing the strings. Must contain at least one
        line.
    """
    def __init__(self, quote_file=None):
        if quote_file is None:
            quote_file = QUOTE_FILE
        self._quote_file = quote_file

    def run_system(self, system):
        """
        Logs a random line from the file passed at initialization.

        Parameters
        ----------
        system
            Not used

        Returns
        -------
        None
        """
        with open(self._quote_file) as handle:
            all_quotes = list(read_quote_file(handle))
        LOGGER.info(random.choice(all_quotes))
