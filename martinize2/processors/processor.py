#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:03:50 2017

@author: peterkroon
"""

class Processor:
    def run_system(self, system):
        mols = []
        for molecule in system.molecules:
            mols.append(self.run_molecule(molecule))
        system.molecules = mols
