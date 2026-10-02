#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 University of Groningen
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
Provides a processor that writes molecule log entries to a logger.
"""

from ..log_helpers import StyleAdapter, get_logger
from .processor import Processor

LOGGER = StyleAdapter(get_logger(__name__))


class OutputLogger(Processor):
    """
    Log all entries stored in ``molecule.log_entries``.

    The processor iterates over all molecules in a system and forwards
    their stored log messages to the configured logger. Any node
    references contained in the formatting arguments are resolved to
    the corresponding molecule nodes before logging.
    """

    def __init__(self):
        self.logger = LOGGER

    def run_system(self, system):
        for molecule in system.molecules:
            for loglevel, entries in molecule.log_entries.items():
                for entry, fmt_args in entries.items():
                    for fmt_arg in fmt_args:
                        fmt_arg = {
                            str(k): molecule.nodes[v]
                            for k, v in fmt_arg.items()
                        }
                        self.logger.log(loglevel, entry, **fmt_arg, type="model")
        return system