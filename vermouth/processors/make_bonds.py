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

from .. import KDTree
from ..molecule import Molecule
from .processor import Processor
from ..utils import distance

import networkx as nx
import numpy as np

COVALENT_RADII = {'H': 0.031, 'C': 0.076, 'N': 0.071, 'O': 0.066, 'S': 0.105}
#VALENCES = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'S': 6}


def bonds_from_distance(system, fudge=1.1):
    system = nx.compose_all(system.molecules)
    idx_to_nodenum = {idx: n for idx, n in enumerate(system)}
    max_dist = max(COVALENT_RADII.values())
    positions = np.array([system.node[n]['position'] for n in system], dtype=float)
    tree = KDTree(positions)
    pairs = tree.query_pairs(2*max_dist*fudge)  # eps=fudge-1?

    for idx1, idx2 in pairs:
        node_idx1 = idx_to_nodenum[idx1]
        node_idx2 = idx_to_nodenum[idx2]
        atom1 = system.node[node_idx1]
        atom2 = system.node[node_idx2]
        element1 = atom1['element']
        element2 = atom2['element']
        if element1 in COVALENT_RADII and element2 in COVALENT_RADII:
            # Elements we do not know never make bonds.
            dist = distance(atom1['position'], atom2['position'])
            bond_distace = COVALENT_RADII[element1] + COVALENT_RADII[element2]
            if dist <= bond_distace * fudge:
                system.add_edge(node_idx1, node_idx2, distance=dist)
    return system


class MakeBonds(Processor):
    def run_system(self, system):
        mols = bonds_from_distance(system)
        system.molecules = list(map(Molecule, (mols.subgraph(mol)
                                               for mol in nx.connected_components(mols))))
        # Restore the force field in each molecule. Setting the force field
        # at the system level propagates it to all the molecules.
        system.force_field = system.force_field
