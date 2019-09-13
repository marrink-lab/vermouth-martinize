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
Contains tests for the NeutralTermini processor.
"""

import pytest
from vermouth.molecule import Molecule
from vermouth.forcefield import get_native_force_field
from vermouth import NeutralTermini


@pytest.fixture
def force_field():
    """
    Martini22 forcefield where modifications have been renamed as if they've
    been produced by the DoMapping processor.
    """
    ff = get_native_force_field('martini22')
    nter = ff.modifications['N-ter'].copy()
    nter.name = (nter.name, )
    cter = ff.modifications['C-ter'].copy()
    cter.name = (cter.name, )
    ff.modifications['N-ter'] = nter
    ff.modifications['C-ter'] = cter
    return ff



@pytest.fixture
def charged_molecule(force_field):
    """
    A molecule with charged termini.
    """
    nodes = [
        (1, {'charge_group': 1, 'resid': 1, 'resname': 'ALA', 'atomname': 'BB',
             'charge': 1, 'atype': 'Qd',
             'modification': force_field.modifications['N-ter'],
             'mapping_weights': {0: 1, 5: 1, 1: 1, 4: 1.0, 2: 1, 3: 1, 6: 1, 7: 1},
             'chain': 'A', 'position': [0.12170435, 0.06658551, -0.0208]}),
        (2, {'charge_group': 2, 'resid': 2, 'resname': 'ALA', 'atomname': 'BB',
             'charge': 0.0, 'atype': 'P4',
             'mapping_weights': {12: 1.0, 17: 1.0, 13: 1.0, 16: 1.0, 14: 1.0, 15: 1.0},
             'chain': 'A', 'position': [0.45269104, 0.23552239, 0.0214209]}),
        (3, {'charge_group': 3, 'resid': 3, 'resname': 'ALA', 'atomname': 'BB',
             'charge': 0.0, 'atype': 'P4',
             'mapping_weights': {22: 1.0, 27: 1.0, 23: 1.0, 26: 1.0, 24: 1.0, 25: 1.0},
             'chain': 'A', 'position': [0.74704179, 0.45218955, -0.0214209]}),
        (4, {'charge_group': 4, 'resid': 4, 'resname': 'ALA', 'atomname': 'BB',
             'charge': 0.0, 'atype': 'P4',
             'mapping_weights': {32: 1.0, 37: 1.0, 33: 1.0, 36: 1.0, 34: 1.0, 35: 1.0},
             'chain': 'A', 'position': [1.07289104, 0.61778657, 0.0214209]}),
        (5, {'charge_group': 5, 'resid': 5, 'resname': 'ALA', 'atomname': 'BB',
             'charge': -1, 'atype': 'Qa',
             'modification': force_field.modifications['C-ter'],
             'mapping_weights': {42: 1.0, 48: 1.0, 43: 1, 47: 1.0, 44: 1, 45: 1, 46: 1},
             'chain': 'A', 'position': [1.40449639, 0.85126265, -0.01729157]})
    ]
    mol = Molecule()
    mol.add_nodes_from(nodes)
    mol.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    return mol


@pytest.fixture
def neutral_molecule(force_field):
    """
    A molecule with neutral termini.
    """
    nodes = [
        (1, {'charge_group': 1, 'resid': 1, 'resname': 'ALA', 'atomname': 'BB',
             'charge': 0, 'atype': 'P5',
             'modification': force_field.modifications['N-ter'],
             'mapping_weights': {0: 1, 5: 1, 1: 1, 4: 1.0, 2: 1, 3: 1, 6: 1, 7: 1},
             'chain': 'A', 'position': [0.12170435, 0.06658551, -0.0208]}),
        (2, {'charge_group': 2, 'resid': 2, 'resname': 'ALA', 'atomname': 'BB',
             'charge': 0.0, 'atype': 'P4',
             'mapping_weights': {12: 1.0, 17: 1.0, 13: 1.0, 16: 1.0, 14: 1.0, 15: 1.0},
             'chain': 'A', 'position': [0.45269104, 0.23552239, 0.0214209]}),
        (3, {'charge_group': 3, 'resid': 3, 'resname': 'ALA', 'atomname': 'BB',
             'charge': 0.0, 'atype': 'P4',
             'mapping_weights': {22: 1.0, 27: 1.0, 23: 1.0, 26: 1.0, 24: 1.0, 25: 1.0},
             'chain': 'A', 'position': [0.74704179, 0.45218955, -0.0214209]}),
        (4, {'charge_group': 4, 'resid': 4, 'resname': 'ALA', 'atomname': 'BB',
             'charge': 0.0, 'atype': 'P4',
             'mapping_weights': {32: 1.0, 37: 1.0, 33: 1.0, 36: 1.0, 34: 1.0, 35: 1.0},
             'chain': 'A', 'position': [1.07289104, 0.61778657, 0.0214209]}),
        (5, {'charge_group': 5, 'resid': 5, 'resname': 'ALA', 'atomname': 'BB',
             'charge': 0, 'atype': 'P5',
             'modification': force_field.modifications['C-ter'],
             'mapping_weights': {42: 1.0, 48: 1.0, 43: 1, 47: 1.0, 44: 1, 45: 1, 46: 1},
             'chain': 'A', 'position': [1.40449639, 0.85126265, -0.01729157]})
    ]
    mol = Molecule()
    mol.add_nodes_from(nodes)
    mol.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    return mol

def test_neutral_termini(charged_molecule, neutral_molecule, force_field):
    """
    Simplest test-case imaginable.
    """
    found = charged_molecule.copy()
    NeutralTermini(force_field).run_molecule(found)
    assert found == neutral_molecule
