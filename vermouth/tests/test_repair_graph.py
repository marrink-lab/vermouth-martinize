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


def build_forcefield_with_mods():
    """
    Build a force field that describe some modifications.

    Returns
    -------
    vermouth.ForceField
    """
    nter = nx.Graph(name='N-ter')
    nter.add_nodes_from((
        (0, {'atomname': 'CA', 'PTM_atom': False, 'element': 'C'}),
        (1, {'atomname': 'N', 'PTM_atom': False, 'element': 'N'}),
        (2, {'atomname': 'HN', 'PTM_atom': False, 'element': 'H'}),
        (3, {'atomname': 'H', 'PTM_atom': True, 'element': 'H'}),
    ))
    nter.add_edges_from([[0, 1], [1, 2], [1, 3]])

    cter = nx.Graph(name='C-ter')
    cter.add_nodes_from((
        (0, {'atomname': 'C', 'PTM_atom': False, 'element': 'C'}),
        (1, {'atomname': 'O', 'PTM_atom': False, 'element': 'O'}),
        (2, {'atomname': 'OXT', 'PTM_atom': True, 'element': 'O'}),
    ))
    cter.add_edges_from([[0, 1], [0, 2]])

    gluh = nx.Graph(name='GLU-H')
    gluh.add_nodes_from((
        (0, {'atomname': 'CD', 'PTM_atom': False, 'element': 'C'}),
        (1, {'atomname': 'OE1', 'PTM_atom': False, 'element': 'O'}),
        (2, {'atomname': 'OE2', 'PTM_atom': False, 'element': 'O'}),
        (3, {'atomname': 'HE1', 'PTM_atom': True, 'element': 'H'}),
    ))
    gluh.add_edges_from([[0, 1], [0, 2], [1, 3]])

    forcefield = copy.copy(vermouth.forcefield.get_native_force_field('universal'))
    forcefield.modifications = [nter, gluh, cter]
    forcefield.renamed_residues[('GLU', ('GLU-H', 'N-ter'))] = 'GLU0'
    return forcefield


def build_system_mod(force_field):
    """
    Build a system with a molecule that comprises modifications.

    Parameters
    ----------
    force_field: vermouth.ForceField
        A force field based on "universal" that desribed the "GLU-H", "C-ter"
        and "N-ter" modifications.
    
    Returns
    -------
    vermouth.System

    See Also
    --------
    build_forcefield_with_mods
    """
    molecule = vermouth.molecule.Molecule()
    # When the name must be fixed, the expected name is stored under 'expected'.
    # If more than one value is acceptable, then 'expected' is a tuple.
    molecule.add_nodes_from((
        # Residue 1 is a N-terminus (atom 14, H) and has a protonated
        # side chain (atom 13, HE1)
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
        # Residue 2 has missing atoms (HA1 and HA2, the hydrogen of the C
        # alpha; HN the hydrogen on the N-ter nitrogen and O, the oxygen at the
        # C-ter)
        (17, {'resid': 2, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'N'}),
        (18, {'resid': 2, 'resname': 'GLY', 'atomname': 'CA', 'chain': 'A', 'element': 'C'}),
        (19, {'resid': 2, 'resname': 'GLY', 'atomname': 'C', 'chain': 'A', 'element': 'C'}),
        # Residue 3 has only N and C specified, everything else is missing.
        (20, {'resid': 3, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'N'}),
        (21, {'resid': 3, 'resname': 'GLY', 'atomname': 'O', 'chain': 'A', 'element': 'O'}),
        # Residue 4 has shuffled and missing names
        (22, {'resid': 4, 'resname': 'GLY', 'atomname': 'CA', 'chain': 'A', 'element': 'N', 'expected': 'N'}),
        (23, {'resid': 4, 'resname': 'GLY', 'chain': 'A', 'element': 'C', 'expected': 'CA'}),
        (24, {'resid': 4, 'resname': 'GLY', 'atomname': 'HA1', 'chain': 'A', 'element': 'C', 'expected': 'C'}),
        (25, {'resid': 4, 'resname': 'GLY', 'atomname': 'HN', 'chain': 'A', 'element': 'O', 'expected': 'O'}),
        (26, {'resid': 4, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'H', 'expected': ('HA1', 'HA2')}),
        (27, {'resid': 4, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'H', 'expected': ('HA1', 'HA2')}),
        (28, {'resid': 4, 'resname': 'GLY', 'atomname': '', 'chain': 'A', 'element': 'H', 'expected': 'HN'}),
        # Residue 5 has a wrong name in a modification, and a wrong name for
        # the anchor.
        (29, {'resid': 5, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'N'}),
        (30, {'resid': 5, 'resname': 'GLY', 'atomname': 'CA', 'chain': 'A', 'element': 'C'}),
        # Should be C
        (31, {'resid': 5, 'resname': 'GLY', 'atomname': 'WRONG', 'chain': 'A', 'element': 'C', 'expected': 'C'}),
        (32, {'resid': 5, 'resname': 'GLY', 'atomname': 'O', 'chain': 'A', 'element': 'O'}),
        (33, {'resid': 5, 'resname': 'GLY', 'atomname': 'HN', 'chain': 'A', 'element': 'H'}),
        (34, {'resid': 5, 'resname': 'GLY', 'atomname': 'HA1', 'chain': 'A', 'element': 'H'}),
        (35, {'resid': 5, 'resname': 'GLY', 'atomname': 'HA2', 'chain': 'A', 'element': 'H'}),
        # Should be OXT
        (36, {'resid': 5, 'resname': 'GLY', 'atomname': 'O2', 'chain': 'A', 'element': 'O', 'expected': 'OXT'}),
    ))
    molecule.add_edges_from([
        [0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [4, 6], [4, 7], [7, 8],
        [7, 9], [7, 10], [10, 11], [10, 12], [12, 13], [14, 0], [15, 1], [16, 0],
        [2, 17], [17, 18], [18, 19],
        [19, 20], [21, 22], [22, 23], [23, 24], [24, 25],
        [22, 28], [23, 26], [23, 27], [24, 29], [29, 30], [30, 31], [31, 32],
        [31, 36], [29, 33], [30, 34], [30, 35],
    ])
    for node in molecule:
        molecule.nodes[node]['atomid'] = node
    system = vermouth.System()
    system.molecules = [molecule]
    system.force_field = force_field
    return system


@pytest.fixture
def forcefield_with_mods():
    return build_forcefield_with_mods()


@pytest.fixture
def system_mod(forcefield_with_mods):
    return build_system_mod(forcefield_with_mods)

@pytest.fixture(params=(True, False))
def repaired_graph(request, system_mod):
    vermouth.RepairGraph(include_graph=request.param).run_system(system_mod)
    return system_mod


@pytest.fixture
def canonicalized_graph(repaired_graph):
    vermouth.CanonicalizeModifications().run_system(repaired_graph)
    return repaired_graph


@pytest.fixture
def renamed_graph(canonicalized_graph):
    vermouth.RenameModifiedResidues().run_system(canonicalized_graph)
    return canonicalized_graph


@pytest.mark.parametrize('node_key', (13,  14, 36))
def test_PTM_atom_true(repaired_graph, node_key):
    molecule = repaired_graph.molecules[0]
    assert molecule.nodes[node_key].get('PTM_atom', False)


@pytest.mark.parametrize('node_key', (
    node_key for node_key in range(16) if node_key not in (13,  14)
))
def test_PTM_atom_false(repaired_graph, node_key):
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


def _list_expected_names():
    molecule = build_system_mod(build_forcefield_with_mods()).molecules[0]
    for key, node in molecule.nodes.items():
        if 'expected' in node:
            expected = node['expected']
        else:
            expected = node['atomname']
        if not isinstance(expected, tuple):
            expected = (expected, )
        yield key, expected


@pytest.mark.parametrize('key, expected_names', _list_expected_names())
def test_name_repaired(repaired_graph, key, expected_names):
    molecule = repaired_graph.molecules[0]
    if not molecule.nodes[key].get('PTM_atom', False):
        print(key, molecule.nodes[key])
        assert molecule.nodes[key]['atomname'] in expected_names


@pytest.mark.parametrize('key, expected_names', _list_expected_names())
def test_name_canonicalized(canonicalized_graph, key, expected_names):
    molecule = canonicalized_graph.molecules[0]
    assert molecule.nodes[key]['atomname'] in expected_names


def test_renaming(renamed_graph):
    for node in renamed_graph.molecules[0].nodes.values():
        if node['resid'] == 1:
            assert node['resname'] == 'GLU0'
        else:
            assert node['resname'] == 'GLY'
