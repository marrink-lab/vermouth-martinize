#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:03:50 2017

@author: peterkroon
"""

class Processor:
    def run_system(self, system):
        for molecule in system.molecules:
            self.run_molecule(molecule)
