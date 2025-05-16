# -*- coding: utf-8 -*-
# Copyright 2025 University of Groningen
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
Unit tests for the StashAttributes processor
"""
from pickle import FALSE

import pytest
from vermouth.tests.datafiles import (
    FF_UNIVERSAL_TEST,
)
from .helper_functions import create_sys_all_attrs
from vermouth.system import System
from vermouth.molecule import Molecule
from vermouth.forcefield import ForceField
from vermouth.processors.stash_attributes import stash_attributes, StashAttributes


@pytest.fixture
def example_mol():
    mol = Molecule(force_field=ForceField(FF_UNIVERSAL_TEST))
    nodes = [
        {'chain': 'A', 'resname': 'GLY', 'resid': 1},  # 0,
        {'chain': 'A', 'resname': 'GLY', 'resid': 2},  # 1,
        {'chain': 'A', 'resname': 'GLY', 'resid': 3},  # 2,
        {'chain': 'A', 'resname': 'PHE', 'resid': 4},  # 3,
        {'chain': 'B', 'resname': 'GLY', 'resid': 1},  # 4,
        {'chain': 'B', 'resname': 'GLY', 'resid': 2},  # 5,
        {'chain': 'B', 'resname': 'GLY', 'resid': 3},  # 6,
        {'chain': 'B', 'resname': 'PHE', 'resid': 4},  # 7,
        {'chain': 'A', 'resname': 'CYS', 'resid': 5},  # 8,
        {'chain': 'C', 'resname': 'not_protein', 'resid': 1}, # 9
    ]
    mol.add_nodes_from(enumerate(nodes))
    mol.add_edges_from([(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (3, 8)])
    return mol

@pytest.mark.parametrize('attributes',
                         (('resid',),
                          ('resid', 'chain')))
def test_stash_attributes(example_mol, attributes):
    stash_attributes(example_mol, attributes)

    for node in example_mol.nodes:
        assert len(example_mol.nodes[node]['stash']) == len(attributes)
        for attr in attributes:
            assert example_mol.nodes[node]['stash'].get(attr) == example_mol.nodes[node].get(attr)


@pytest.mark.parametrize('do_twice, expected',
                         ((False, False),
                          (True, True)
                          ))
def test_already_stored(example_mol, caplog, do_twice, expected):
    stash_attributes(example_mol, ('resid',))

    if do_twice:
        stash_attributes(example_mol, ('resid',))

    assert any([rec.levelname == 'WARNING' for rec in caplog.records]) == expected

def test_StashAttributes(example_mol):

    system = System()
    system.molecules.append(example_mol)

    processor = StashAttributes(attributes=('resid',))
    processor.run_system(system)

    for molecule in system.molecules:
        for node in molecule.nodes:
            assert molecule.nodes[node]['stash']['resid'] == molecule.nodes[node]['resid']

