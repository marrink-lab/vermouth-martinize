# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:48:46 2017

@author: Peter Kroon
"""

from martinize2 import *


PATH = '../molecules/cycliclipopeptide_2.pdb'
CG_graph = martinize(PATH, False)

write_pdb(CG_graph, "test.pdb", conect=True)
