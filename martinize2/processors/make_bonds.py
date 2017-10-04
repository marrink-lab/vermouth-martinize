#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:48:58 2017

@author: peterkroon
"""

from .processor import Processor

import networkx as nx
import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
    from scipy.spatial.distance import euclidean, cdist
except ImportError:
    print('Using redistributed KDTree')
    from ..kdtree import KDTree

    def euclidean(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))


COVALENT_RADII = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05}
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
        distance = euclidean(atom1['position'], atom2['position'])
        bond_distace = COVALENT_RADII[atom1['element']] + COVALENT_RADII[atom2['element']]
        if distance <= bond_distace * fudge:
            system.add_edge(node_idx1, node_idx2, distance=distance)
    return system

class MakeBonds(Processor):
    def run_system(self, system):
        mols = bonds_from_distance(system)
        system.molecules = list(nx.connected_component_subgraphs(mols))
#        system.molecules = [mols]
