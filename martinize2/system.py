#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
