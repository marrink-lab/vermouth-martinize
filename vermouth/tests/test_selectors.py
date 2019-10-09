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

import functools

import pytest

import vermouth
from vermouth.pdb.pdb import read_pdb
from vermouth.tests.datafiles import (
    PDB_PROTEIN,
    PDB_NOT_PROTEIN,
    PDB_PARTIALLY_PROTEIN,
)


@pytest.mark.parametrize(
    'molecules, reference_answer',
    [(read_pdb(str(path)), answer) for path, answer in [
        (PDB_PROTEIN, True),
        (PDB_NOT_PROTEIN, False),
        (PDB_PARTIALLY_PROTEIN, False),
    ]]
)
def test_is_protein(molecules, reference_answer):
    """
    Make sure that proteins are correctly identified as such.
    """
    assert len(molecules) == 1
    molecule = molecules[0]
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


@pytest.mark.parametrize('node', (
    {},
    {'resname': 'PLOP', 'atomname': 'A'},
))
def test_select_all(node):
    """
    Test that :func:`vermouth.selectors.select_all` indeed accept every nodes.
    """
    assert vermouth.selectors.select_all(node)


@pytest.mark.parametrize('node, expected', (
    ({}, False),  # empty node
    ({'atomname': 'not BB'}, False),  # wrong atomname
    ({'resname': 'anything'}, False),  # no atomname
    ({'atomname': 'BB'}, True),  # only correct atomname
    ({'atomname': 'BB', 'other': 'something'}, True),  # correct atomname + other attribute
))
def test_select_backbone(node, expected):
    """
    Test that :func:`vermouth.selectors.select_backbone` only select beads with
    'BB' as atom name.
    """
    assert vermouth.selectors.select_backbone(node) == expected


@pytest.mark.parametrize('node, attribute, values, expected', (
    ({}, 'name', ['something', 'other'], False),
    ({'name': 'nothing'}, 'name', ['something', 'other'], False),
    ({'name': 'something'}, 'name', ['something', 'other'], True),
    ({'name': 'nothing', 'plop': 'other'}, 'name', ['something', 'other'], False),
    ({'name': 'nothing'}, 'name', [], False),
))
def test_select_proto_select_attribute_in(node, attribute, values, expected):
    """
    Test that :func:`vermouth.selectors.proto_select_attribute_in` works as
    expected and can be used with :func:`functools.partial`.
    """
    # The use of functools.partial is not necessary. Though this is how
    # proto_select_attribute_in is used in the example from its docstring so
    # it should work.
    select_attribute_in = functools.partial(
        vermouth.selectors.proto_select_attribute_in,
        attribute=attribute, values=values,
    )
    assert select_attribute_in(node) == expected


@pytest.mark.parametrize('node, templates, ignore_keys, expected', (
    ({}, [], (), False),  # Everything empty
    ({}, [{'name': 'A'}, {'name': 'B'}], (), False),  # Empty node
    ({'name': 'A'}, [{'resname': 'A'}, {'resid': 1}], (), False),  # Not matching
    ({'name': 'A'}, [{'name': 'not A'}, {'name': 'A', 'resname': 'something'}], (), False),
    ({'name': 'A'}, [{'resid': 1, 'resname': 'something'}, {'name': 'A'}], (), True),
    ({'name': 'A', 'resname': 'plop'}, [{'name': 'A'}], ('resname', 'other'), True),
))
def test_proto_multi_templates(node, templates, ignore_keys, expected):
    """
    Test that :func:`vermouth.selectors.proto_multi_templates` works as
    expected and can be used with :func:`functools.partial`.
    """
    multi_templates = functools.partial(
        vermouth.selectors.proto_multi_templates,
        templates=templates,
        ignore_keys=ignore_keys,
    )
    assert multi_templates(node) == expected


def test_filter_minimal():
    """
    Test that :func:`vermouth.selectors.filter_minimal` works as expected.
    """
    # Build a molecule that has all even atoms with no positions.
    molecules = read_pdb(str(PDB_PROTEIN))
    assert len(molecules) == 1
    molecule = molecules[0]

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
