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
Tests for the AddMoleculeEdgesAtDistance processor.
"""

import pytest

import vermouth
from vermouth.molecule import Choice
from vermouth.pdb.pdb import read_pdb
from .datafiles import SHORT_DNA

DNA_DONNORS = [
    {'resname': Choice(['DA', 'DA3', 'DA5']), 'atomname': Choice(['C2', 'N6'])},
    {'resname': Choice(['DG', 'DG3', 'DG5']), 'atomname': Choice(['N1', 'N2'])},
    {'resname': Choice(['DC', 'DC3', 'DC5']), 'atomname': 'N4'},
    {'resname': Choice(['DT', 'DT3', 'DT5']), 'atomname': 'N3'},
]
DNA_ACCEPTORS = [
    {'resname': Choice(['DA', 'DA3', 'DA5']), 'atomname': 'N1'},
    {'resname': Choice(['DG', 'DG3', 'DG5']), 'atomname': 'O6'},
    {'resname': Choice(['DC', 'DC3', 'DC5']), 'atomname': Choice(['N3', 'O2'])},
    {'resname': Choice(['DT', 'DT3', 'DT5']), 'atomname': Choice(['O2', 'O4'])},
]
DNA_HB_DIST = 0.3


@pytest.fixture
def short_dna():
    """
    Build a system that contains a short DNA double strand and add hydrogen
    bonds using :class:`vermouth.AddMoleculeEdgesAtDistance`.
    """
    molecule = read_pdb(SHORT_DNA)
    assert len(molecule.edges) == 0

    system = vermouth.system.System()
    system.molecules = [molecule]

    processor = vermouth.AddMoleculeEdgesAtDistance(
        DNA_HB_DIST, DNA_DONNORS, DNA_ACCEPTORS
    )
    system = processor.run_system(system)
    return system


def test_add_molecule_edges_distance(short_dna):
    """
    Assure that :class:`vermouth.AddMoleculeEdgesAtDistance` adds the expected
    edges.
    """
    expected = set([
        (52, 74),
        (10, 117),
        (54, 73),
        (13, 114),
        (35, 92),
        (32, 95),
        (51, 76),
        (33, 93),
        (11, 115),
    ])
    assert set(short_dna.molecules[0].edges) == expected
