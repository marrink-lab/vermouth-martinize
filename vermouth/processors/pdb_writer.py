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

from . import Processor
from ..pdb import write_pdb


class PDBWriter(Processor):
    def __init__(self, path, conect=True, omit_charges=True, nan_missing_pos=False, defer_writing=True):
        self.path = path
        self.conect = conect
        self.omit_charges = omit_charges
        self.nan_missing_pos = nan_missing_pos
        self.defer_writing = defer_writing

    def run_system(self, system):
        write_pdb(system, path=self.path, conect=self.conect,
                  omit_charges=self.omit_charges, nan_missing_pos=self.nan_missing_pos,
                  defer_writing=self.defer_writing)
        return system
