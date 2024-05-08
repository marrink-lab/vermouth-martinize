#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 University of Groningen
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

from .processor import Processor
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


class PrintMolLogs(Processor):
    def __init__(self, logtype='model'):
        self.logtype = logtype

    def run_molecule(self, molecule):
        for loglevel, entries in molecule.log_entries.items():
            for entry, fmt_args in entries.items():
                for fmt_arg in fmt_args:
                    fmt_arg = {str(k): molecule.nodes[v] for k, v in fmt_arg.items()}
                    LOGGER.log(loglevel, entry, **fmt_arg, type=self.logtype)
        return molecule
