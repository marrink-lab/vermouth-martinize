#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:41:10 2017

@author: peterkroon
"""

from ..pdb import read_pdb
from .processor import Processor

class PDBInput(Processor):
    def run_system(self, system, filename):
        molecule = read_pdb(filename)
        system.add_molecule(molecule)
