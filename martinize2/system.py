#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:39:09 2017

@author: peterkroon
"""


class System:
    def __init__(self):
        self.molecules = []

    def add_molecule(self, molecule):
        self.molecules.append(molecule)

    @property
    def num_particles(self):
        return sum(len(mol) for mol in self.molecules)
