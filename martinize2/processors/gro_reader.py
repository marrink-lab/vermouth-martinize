#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:41:10 2017

@author: peterkroon
"""

from ..gmx import gro
from .processor import Processor

class GROInput(Processor):
    def run_system(self, system, filename):
        molecule = gro.read_gro(filename)
        system.add_molecule(molecule)
