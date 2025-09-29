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

import random

from .processor import Processor
from ..data import QUOTES
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


class Quoter(Processor):
    """
    Processor that can produce random string taken from a list. Useful for e.g.
    quotes.

    Parameters
    ----------
    quotes: list[str]
        List of strings describing the quotes.
    """
    def __init__(self, quotes=None):
        if not quotes:
            quotes = QUOTES
        self.quotes = quotes

    def run_system(self, system):
        """
        Logs a random line from the list passed at initialization.

        Parameters
        ----------
        system
            Not used

        Returns
        -------
        None
        """
        LOGGER.info(random.choice(self.quotes))
