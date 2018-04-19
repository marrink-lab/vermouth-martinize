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
Test graph reparation and related operations.
"""

import pytest
import vermouth
import vermouth.forcefield
import networkx as nx
import copy


@pytest.fixture
def forcefield_with_mods():
    nter = nx.Graph(name='N-ter')
    nter.add_nodes_from((
        (0, {'atomname': 'CA', 'PTM_atom': False, 'element': 'C'}),
        (1, {'atomname': 'N', 'PTM_atom': False, 'element': 'N'}),
        (2, {'atomname': 'HN', 'PTM_atom': False, 'element': 'H'}),
        (3, {'atomname': 'H', 'PTM_atom': True, 'element': 'H'}),
    ))
    nter.add_edges_from([[0, 1], [1, 2], [1, 3]])

    gluh = nx.Graph(name='GLU-H')
    gluh.add_nodes_from((
        (0, {'atomname': 'CD', 'PTM_atom': False, 'element': 'C'}),
        (1, {'atomname': 'OE1', 'PTM_atom': False, 'element': 'O'}),
        (2, {'atomname': 'OE2', 'PTM_atom': False, 'element': 'O'}),
        (3, {'atomname': 'HE1', 'PTM_atom': True, 'element': 'H'}),
    ))
    gluh.add_edges_from([[0, 1], [0, 2], [1, 3]])
    forcefield = copy.copy(vermouth.forcefield.FORCE_FIELDS['universal'])
    forcefield.modifications = [nter, gluh]
    return forcefield


@pytest.fixture
def system_mod(forcefield_with_mods):
    molecule = vermouth.molecule.Molecule()
    molecule.add_nodes_from((
        (0, {'resid': 1, 'resname': 'GLU', 'atomname': 'N', 'chain': 'A', 'element': 'N'}),
        (1, {'resid': 1, 'resname': 'GLU', 'atomname': 'CA', 'chain': 'A', 'element': 'C'}),
        (2, {'resid': 1, 'resname': 'GLU', 'atomname': 'C', 'chain': 'A', 'element': 'C'}),
        (3, {'resid': 1, 'resname': 'GLU', 'atomname': 'O', 'chain': 'A', 'element': 'O'}),
        (4, {'resid': 1, 'resname': 'GLU', 'atomname': 'CB', 'chain': 'A', 'element': 'C'}),
        (5, {'resid': 1, 'resname': 'GLU', 'atomname': 'HB1', 'chain': 'A', 'element': 'H'}),
        (6, {'resid': 1, 'resname': 'GLU', 'atomname': 'HB2', 'chain': 'A', 'element': 'H'}),
        (7, {'resid': 1, 'resname': 'GLU', 'atomname': 'CG', 'chain': 'A', 'element': 'C'}),
        (8, {'resid': 1, 'resname': 'GLU', 'atomname': 'HG1', 'chain': 'A', 'element': 'H'}),
        (9, {'resid': 1, 'resname': 'GLU', 'atomname': 'HG2', 'chain': 'A', 'element': 'H'}),
        (10, {'resid': 1, 'resname': 'GLU', 'atomname': 'CD', 'chain': 'A', 'element': 'C'}),
        (11, {'resid': 1, 'resname': 'GLU', 'atomname': 'OE2', 'chain': 'A', 'element': 'O'}),
        (12, {'resid': 1, 'resname': 'GLU', 'atomname': 'OE1', 'chain': 'A', 'element': 'O'}),
        (13, {'resid': 1, 'resname': 'GLU', 'atomname': 'HE1', 'chain': 'A', 'element': 'H'}),
        (14, {'resid': 1, 'resname': 'GLU', 'atomname': 'H', 'chain': 'A', 'element': 'H'}),
        (15, {'resid': 1, 'resname': 'GLU', 'atomname': 'HA', 'chain': 'A', 'element': 'H'}),
        (16, {'resid': 1, 'resname': 'GLU', 'atomname': 'HN', 'chain': 'A', 'element': 'H'}),
    ))
    molecule.add_edges_from([
        [0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [4, 6], [4, 7], [7, 8],
        [7, 9], [7, 10], [10, 11], [10, 12], [12, 13], [14, 0], [15, 1], [16, 0],
    ])
    system = vermouth.System()
    system.molecules = [molecule]
    system.force_field = forcefield_with_mods
    return system


@pytest.fixture
def repaired_graph(system_mod):
    vermouth.RepairGraph().run_system(system_mod)
    return system_mod


@pytest.fixture
def canonicalized_graph(repaired_graph):
    vermouth.CanonicalizeModifications().run_system(repaired_graph)
    return repaired_graph


@pytest.mark.parametrize('node_key', (13,  14))
def test_PTM_atom_true(repaired_graph, node_key):
    assert repaired_graph[node_key].get('PTM_atom', False)


@pytest.mark.parametrize('node_key', (
    node_key for node_key in range(16) if node_key not in (13,  14)
))
def test_PTM_atom_true(repaired_graph, node_key):
    molecule = repaired_graph.molecules[0]
    assert not molecule.nodes[node_key].get('PTM_atom', False)


def test_uniq_names_repaired(repaired_graph):
    assert len(repaired_graph.molecules) == 1
    molecule = repaired_graph.molecules[0]
    atoms = [(node['resid'], node['atomname'])
             for node in molecule.nodes.values()]
    deduplicated = set(atoms)
    assert len(atoms) == len(deduplicated)


def test_uniq_names_canonicalize(canonicalized_graph):
    assert len(canonicalized_graph.molecules) == 1
    molecule = canonicalized_graph.molecules[0]
    atoms = [(node['resid'], node['atomname'])
             for node in molecule.nodes.values()]
    deduplicated = set(atoms)
    assert len(atoms) == len(deduplicated)
