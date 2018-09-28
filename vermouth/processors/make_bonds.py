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
"""
Provides a processor that can add edges to a graph based on geometric criteria.
"""


import networkx as nx
import numpy as np

from .. import KDTree
from ..molecule import Molecule
from .processor import Processor
from ..utils import distance

# Van der Waals radii from A. Bondi, J. Phys. Chem., 68, 441-452, 1964.
# https://doi.org/10.1021/j100785a001
# For hydrogen, we use R.S. Rowland & R. Taylor, J.Phys.Chem., 100, 7384-7391, 1996.
# https://doi.org/10.1021/jp953141
VDW_RADII = {  # in nm
    'H': 0.120,
    'He': 0.140,
    'C': 0.170,
    'N': 0.155,
    'O': 0.152,
    'F': 0.147,
    'Ne': 0.154,
    'Si': 0.210,
    'P': 0.180,
    'S': 0.180,
    'Cl': 0.175,
    'Ar': 0.188,
    'As': 0.185,
    'Se': 1.90,
    'Br': 0.185,
    'Kr': 0.202,
    'Te': 0.206,
    'I': 0.198,
    'Xe': 0.216,
}
#VALENCES = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'S': 6}


def bonds_from_distance(system, fudge=1.2):
    """
    Creates edges between nodes of molecules in system based on a distance
    criterion. Nodes in system must have `position` and `element` attributes.
    The possible distance between nodes is determined by values in
    `VDW_RADII`.

    Notes
    -----
    Elements that are not in `VDW_RADII` do not make bonds.

    Parameters
    ----------
    system: :class:`~vermouth.system.System`
        The system in which to add edges.
    fudge: :class:`~numbers.Number`
        Increase the allowed distance by this factor.

    Returns
    -------
    :class:`networkx.Graph`
        A new graph where edges are added between nodes that are within a
        certain distance from each other. It is probably disconnected.
    """
    system = nx.compose_all(system.molecules)
    # We filter out the nodes for which we do not know the radius. Indeed, we
    # consider these nodes cannot make bonds. The filtering is done before we
    # enter the KDTree; we only provide to the KDTree the position of the nodes
    # that could make a bond. `idx_to_nodenum` make the link between the
    # indices in the `positions` array, and the node keys in the `system`
    # graph.
    idx_to_nodenum = {
        idx: n
        for idx, n in enumerate(
                subn
                for subn in system
                if system.nodes[subn].get('element') in VDW_RADII
        )
    }
    max_dist = max(
        VDW_RADII[node.get('element')]
        for node in system.nodes.values()
        if node.get('element') in VDW_RADII
    )
    positions = np.array([
        node['position']
        for node in system.nodes.values()
        if node.get('element') in VDW_RADII
    ], dtype=float)
    tree = KDTree(positions)
    pairs = tree.sparse_distance_matrix(tree, max_dist * fudge)

    for (idx1, idx2), dist in pairs.items():
        if idx1 >= idx2:
            continue
        node_idx1 = idx_to_nodenum[idx1]
        node_idx2 = idx_to_nodenum[idx2]
        atom1 = system.node[node_idx1]
        atom2 = system.node[node_idx2]
        element1 = atom1['element']
        element2 = atom2['element']

        bond_distance = 0.5 * (VDW_RADII[element1] + VDW_RADII[element2])
        if dist <= bond_distance * fudge:
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
