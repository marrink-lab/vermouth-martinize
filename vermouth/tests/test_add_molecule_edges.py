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

# The redefined-outer-name check from pylint wrongly catches the use of pytest
# fixtures.
# pylint: disable=redefined-outer-name

import pytest

import vermouth
from vermouth.pdb.pdb import read_pdb
from vermouth.processors.add_molecule_edges import (
    DNA_ACCEPTORS, DNA_DONORS, DNA_HB_DIST,
)
from .datafiles import SHORT_DNA


@pytest.fixture
def short_dna():
    """
    Build a system that contains a short DNA double strand without edges.
    """
    molecules = read_pdb(SHORT_DNA)

    system = vermouth.system.System()
    system.molecules = molecules
    return system


def short_dna_general(short_dna):
    """
    DNA double strands with hydrogen bonds added using
    :class:`vermouth.AddMoleculeEdgesAtDistance`.
    """
    processor = vermouth.AddMoleculeEdgesAtDistance(
        DNA_HB_DIST, DNA_DONORS, DNA_ACCEPTORS
    )
    system = processor.run_system(short_dna)
    return system


def short_dna_strands(short_dna):
    """
    DNA double strands with hydrogen bonds added using
    :class:`vermouth.MergeNucleicStrands`.
    """
    processor = vermouth.MergeNucleicStrands()
    system = processor.run_system(short_dna)
    return system


@pytest.fixture(params=(short_dna_general, short_dna_strands))
def short_dna_edges(request, short_dna):
    """
    Create successively a system with edges produced by
    :class:`vermouth.AddMoleculeEdgesAtDistance` and by
    :class:`vermouth.MergeNucleicStrands`. Having this fixture allows to run
    :func:`test_add_molecule_edges_distance` with the output of both
    :func:`short_dna_general` and :func:`short_dna_strands` the outputs of
    which should be identical.
    """
    return request.param(short_dna)


def test_add_molecule_edges_distance(short_dna_edges):
    """
    Assure that :class:`vermouth.AddMoleculeEdgesAtDistance` and
    :class:`vermouth.MergeNucleicStrands` adds the expected edges.
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
        (73, 74),
        (32, 33),
        (52, 54),
        (93, 95),
        (114, 115),
        (11, 13),
    ])
    assert set(short_dna_edges.molecules[0].edges) == expected
