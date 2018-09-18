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
Tests for the selectors in :mod:`vermouth.selectors`.
"""

import pytest
import vermouth
from vermouth.pdb.pdb import read_pdb
from vermouth.tests.datafiles import (
    PDB_PROTEIN,
    PDB_NOT_PROTEIN,
    PDB_PARTIALLY_PROTEIN,
)


@pytest.mark.parametrize(
    'molecule, reference_answer',
    [(read_pdb(str(path)), answer) for path, answer in [
        (PDB_PROTEIN, True),
        (PDB_NOT_PROTEIN, False),
        (PDB_PARTIALLY_PROTEIN, False),
    ]]
)
def test_is_protein(molecule, reference_answer):
    """
    Make sure that proteins are correctly identified as such.
    """
    assert vermouth.selectors.is_protein(molecule) == reference_answer


@pytest.mark.parametrize(
    'atom, reference_answer',
    (
        ({'position': [0, 0, 0]}, True),
        ({'position': None}, False),
        ({}, False),
    )
)
def test_selector_no_position(atom, reference_answer):
    """
    Test that :func:`vermouth.selectors.selector_has_position` works as expected.
    """
    assert vermouth.selectors.selector_has_position(atom) == reference_answer


def test_filter_minimal():
    """
    Test that :func:`vermouth.selectors.filter_minimal` works as expected.
    """
    # Build a molecule that has all even atoms with no positions.
    molecule = read_pdb(str(PDB_PROTEIN))
    for atom in list(molecule.nodes.values())[::2]:
        atom['position'] = None
    # This means that we want to keep the odd atoms
    to_keep = list(molecule.nodes)[1::2]

    filtered = vermouth.selectors.filter_minimal(
        molecule,
        selector=vermouth.selectors.selector_has_position
    )

    # Do we keep the right atoms?
    assert list(filtered) == to_keep
