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
Created on Wed Oct  4 10:39:09 2017

@author: peterkroon
"""


class System:
    def __init__(self):
        self.molecules = []
        self._force_field = None

    @property
    def force_field(self):
        return self._force_field

    @force_field.setter
    def force_field(self, value):
        self._force_field = value
        for molecule in self.molecules:
            molecule._force_field = value

    def add_molecule(self, molecule):
        self.molecules.append(molecule)

    @property
    def num_particles(self):
        return sum(len(mol) for mol in self.molecules)

    def copy(self):
        new_system = self.__class__()
        new_system._force_field = self.force_field
        new_system.molecules = [mol.copy() for mol in self.molecules]
        return new_system
