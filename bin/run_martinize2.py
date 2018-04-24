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
Created on Tue Aug 22 11:48:46 2017

@author: Peter Kroon
"""
#import matplotlib.pyplot as plt
#plt.close('all')
from vermouth import *

import os.path as osp

import numpy as np

#PATH = '../molecules/cycliclipopeptide_2.pdb'
#PATH = '../molecules/cyclicpeptide_2.pdb'
PATH = '../molecules/glkfk.pdb'
#PATH = '../molecules/6-macro-8_cartwheel.gro'
#PATH = '../molecules/6-macro-16.gro'
#PATH = '../molecules/6-macro-16-rtc-eq-nodisre.pdb'
#PATH = '../molecules/3-macro-1.gro'
#
#write_pdb(CG_graph, "6-macro-16-rtc-eq-nodisre-CG.pdb", conect=True)

system = System()

ext = osp.splitext(PATH)[-1]

if ext.casefold() == '.pdb':
    PDBInput().run_system(system, PATH)
elif ext.casefold() == '.gro':
    GROInput().run_system(system, PATH)
else:
    raise RuntimeError
MakeBonds().run_system(system)
RepairGraph().run_system(system)
CanonizePTMs().run_system(system)
DoMapping().run_system(system)
ApplyBlocks().run_system(system)
DoLinks().run_system(system)

print(system)


for mol in system.molecules:
    for idx in list(mol):
        if 'position' not in mol.nodes[idx]:
            node = mol.nodes[idx]
            print(node['resname'], node['resid'], node['atomname'])
#            mol.nodes[idx]['position'] = np.array([np.nan, np.nan, np.nan])
            mol.remove_node(idx)
            
    draw(mol, node_size=30, node_color=tuple(np.random.rand(3)), with_label=True)

show()
