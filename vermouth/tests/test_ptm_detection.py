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
Test PTM detection and canonicalisation.
"""

import networkx as nx
import pytest

import vermouth
import vermouth.processors.canonicalize_modifications as canmod

# pylint: disable=redefined-outer-name


@pytest.fixture
def known_ptm_graphs():
    """
    Provide some known PTMs in a defined order
    """
    ptm_graphs = []

    nh_ptm = nx.Graph(name='NH')
    nh_ptm.add_nodes_from([
        (0, {'atomname': 'H', 'PTM_atom': True, 'element': 'H'}),
        (1, {'atomname': 'N', 'PTM_atom': False, 'element': 'N'}),
    ])
    nh_ptm.add_edge(0, 1)
    ptm_graphs.append(nh_ptm)

    cooc_ptm = nx.Graph(name='COOC')
    cooc_ptm.add_nodes_from([
        (0, {'atomname': 'C', 'PTM_atom': False, 'element': 'C'}),
        (1, {'atomname': 'O', 'PTM_atom': True, 'element': 'O'}),
        (2, {'atomname': 'O', 'PTM_atom': True, 'element': 'O'}),
        (3, {'atomname': 'C', 'PTM_atom': False, 'element': 'C'}),
    ])
    cooc_ptm.add_edges_from([(0, 1), (1, 2), (2, 3)])
    ptm_graphs.append(cooc_ptm)

    return sorted(ptm_graphs, key=len, reverse=True)


def make_molecule(atoms, edges):
    """
    Makes molecules from atoms and edges
    """
    mol = vermouth.molecule.Molecule()
    mol.add_nodes_from(atoms.items())
    mol.add_edges_from(edges)
    return mol


@pytest.mark.parametrize('atoms, edges, expected', [
    ({}, [], []),
    (
        {
            0: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 1},
            1: {'atomname': 'H', 'PTM_atom': True, 'element': 'H', 'resid': 1},
            2: {'atomname': 'N', 'PTM_atom': True, 'element': 'N', 'resid': 1},
        },
        [(0, 1), (1, 2)],
        [({1, 2}, {0})]
    ),
    (
        {
            0: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 1},
            1: {'atomname': 'H', 'PTM_atom': False, 'element': 'H', 'resid': 1},
            2: {'atomname': 'N', 'PTM_atom': True, 'element': 'N', 'resid': 1},
        },
        [(0, 1), (1, 2)],
        [({2}, {1})]
    ),
    (
        {
            0: {'atomname': 'N', 'PTM_atom': True, 'element': 'N', 'resid': 1},
            1: {'atomname': 'H', 'PTM_atom': False, 'element': 'H', 'resid': 1},
            2: {'atomname': 'N', 'PTM_atom': True, 'element': 'N', 'resid': 1},
        },
        [(0, 1), (1, 2)],
        [({0}, {1}), ({2}, {1})]
    ),
    (
        {
            0: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 1},
            1: {'atomname': 'H', 'PTM_atom': True, 'element': 'H', 'resid': 1},
            2: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 2},
        },
        [(0, 1), (1, 2)],
        [({1}, {0, 2})]
    ),
    (
        {
            0: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 1},
            1: {'atomname': 'H', 'PTM_atom': True, 'element': 'H', 'resid': 1},
            2: {'atomname': 'N', 'PTM_atom': True, 'element': 'N', 'resid': 2},
            3: {'atomname': 'CA', 'PTM_atom': False, 'element': 'N', 'resid': 2},
        },
        [(0, 1), (1, 2), (2, 3)],
        [({1, 2}, {0, 3})]
    ),
])
def test_ptm_groups(atoms, edges, expected):
    """
    Make sure PTM atoms are grouped correctly with appropriate anchors
    """
    molecule = make_molecule(atoms, edges)

    found = canmod.find_ptm_atoms(molecule)
    assert expected == found


@pytest.mark.parametrize('atoms, edges, expected', [
    pytest.param(
        # This needs to raise a KeyError, because not all the anchors are
        # covered. This is the root of #140
        {
            0: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 1},
            1: {'atomname': 'H', 'PTM_atom': True, 'element': 'H', 'resid': 1},
            2: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 2},
        },
        [(0, 1), (1, 2)],
        [('NH', {1: 0, 2: 1})],
        marks=pytest.mark.xfail(raises=KeyError, strict=True)
    ),
    (
        # Simplest case: one PTM atom for 1 residue
        {
            0: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 1},
            1: {'atomname': 'H', 'PTM_atom': True, 'element': 'H', 'resid': 1},
        },
        [(0, 1)],
        [('NH', {0: 1, 1: 0})]
    ),
    (
        # Two PTM atoms with a shared anchor
        {
            0: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 1},
            1: {'atomname': 'H', 'PTM_atom': True, 'element': 'H', 'resid': 1},
            2: {'atomname': 'H', 'PTM_atom': True, 'element': 'H', 'resid': 1},
        },
        [(0, 1), (0, 2)],
        [('NH', {0: 1, 1: 0}), ('NH', {0: 1, 2: 0})]
    ),
    (
        # Two PTM atoms with two anchors covered (?) by 2 fragments
        {
            0: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 1},
            1: {'atomname': 'H', 'PTM_atom': True, 'element': 'H', 'resid': 1},
            2: {'atomname': 'H', 'PTM_atom': True, 'element': 'H', 'resid': 2},
            3: {'atomname': 'N', 'PTM_atom': False, 'element': 'N', 'resid': 2},
        },
        [(0, 1), (1, 2), (2, 3)],
        [('NH', {0: 1, 1: 0}), ('NH', {2: 0, 3: 1})]
    ),
    (
        # Two PTM atoms with two anchors covered by 1 fragment
        {
            0: {'atomname': 'C', 'PTM_atom': False, 'element': 'C', 'resid': 1},
            1: {'atomname': 'O', 'PTM_atom': True, 'element': 'O', 'resid': 1},
            2: {'atomname': 'O', 'PTM_atom': True, 'element': 'O', 'resid': 2},
            3: {'atomname': 'C', 'PTM_atom': False, 'element': 'C', 'resid': 2},
        },
        [(0, 1), (1, 2), (2, 3)],
        [('COOC', {0: 0, 1: 1, 2: 2, 3: 3})]
    ),
])
def test_identify_ptms(known_ptm_graphs, atoms, edges, expected):
    """
    Make sure PTMs are identified correctly.
    """
    molecule = make_molecule(atoms, edges)

    ptms = canmod.find_ptm_atoms(molecule)
    known_ptms = [(ptm_graph, canmod.PTMGraphMatcher(molecule, ptm_graph))
                  for ptm_graph in known_ptm_graphs]

    found = canmod.identify_ptms(molecule, ptms, known_ptms)
    found = [(ptm.name, match) for ptm, match in found]
    assert found == expected
