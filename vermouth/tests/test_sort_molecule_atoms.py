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
Tests for the:class:`vermouth.processors.sort_molecule_atoms.SortMoleculeAtoms`
processor.
"""

import copy
from vermouth import Molecule, SortMoleculeAtoms
from vermouth.molecule import Interaction


def test_sort_molecule_atoms():
    """
    Test the :class:`vermouth.processors.sort_molecule_atoms.SortMoleculeAtoms`
    processor in normal conditions.
    """
    molecule = Molecule()
    nodes = [
        (6, {'chain': 'C', 'resid': 2, 'resname': 'AA0', 'atomname': 'A5'}),
        (5, {'chain': 'A', 'resid': 2, 'resname': 'XX0', 'atomname': 'A0'}),
        (4, {'chain': 'B', 'resid': 1, 'resname': 'XX1', 'atomname': 'A2'}),
        (3, {'chain': 'C', 'resid': 1, 'resname': 'XX0', 'atomname': 'A4'}),
        (2, {'chain': 'C', 'resid': 2, 'resname': 'AA1', 'atomname': 'A6'}),
        (1, {'chain': 'A', 'resid': 2, 'resname': 'XX1', 'atomname': 'A1'}),
        (0, {'chain': 'B', 'resid': 2, 'resname': 'XX0', 'atomname': 'A3'}),
    ]
    edges = [
        (0, 6, {'content': 'something'}),
        (1, 2, {'hello': 34}),
    ]
    interactions = {
        'bonds': [
            Interaction(atoms=[0, 1], parameters=['a', 'b'], meta={'a': 7}),
            Interaction(atoms=[3, 4], parameters=['c', 'd'], meta={'b': 5}),
        ],
        'angles': [
            Interaction(atoms=[0, 1, 2], parameters=['b', 'c'], meta={'a': 0}),
        ],
    }

    molecule.add_nodes_from(nodes)
    molecule.add_edges_from(edges)
    molecule.interactions = copy.copy(interactions)

    processor = SortMoleculeAtoms()
    processor.run_molecule(molecule)

    assert list(molecule.nodes) == [5, 1, 4, 0, 3, 6, 2]
    assert sorted(molecule.edges(data=True)) == edges
    assert molecule.interactions == interactions
