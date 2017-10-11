# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:48:46 2017

@author: Peter Kroon
"""
#import matplotlib.pyplot as plt
#plt.close('all')
from martinize2 import *

import numpy as np

PATH = '../molecules/cycliclipopeptide_2.pdb'
#PATH = '../molecules/glkfk.pdb'
#PATH = '../molecules/6-macro-8_cartwheel.gro'
#PATH = '../molecules/6-macro-16.gro'
#PATH = '../molecules/6-macro-16-rtc-eq-nodisre.pdb'
#CG_graph = martinize(PATH, True)
#
#write_pdb(CG_graph, "6-macro-16-rtc-eq-nodisre-CG.pdb", conect=True)

system = System()

GROInput().run_system(system, PATH)
MakeBonds().run_system(system)
RepairGraph().run_system(system)
DoMapping().run_system(system)

print(system)


for mol in system.molecules:
#    for idx in mol:
#        print(mol.nodes[idx]['position'])
    draw(mol, node_size=30, node_color=tuple(np.random.rand(3)))

show()
