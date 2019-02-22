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
Test the :class:`~vermouth.processors.set_molecule_meta.SetMoleculeMeta` processor.
"""

import copy

import pytest

from vermouth import System
from vermouth.molecule import Molecule
from vermouth.processors.set_molecule_meta import SetMoleculeMeta
from vermouth.utils import are_different


@pytest.mark.parametrize('prior, meta, expected', (
    (
        {'foo': 'bar', 'toto': [3, 4, None]},
        {'new key': 9.0},
        {'foo': 'bar', 'toto': [3, 4, None], 'new key': 9.0},
    ),
    (
        {'foo': 'bar', 'toto': [3, 4, None]},
        {'foo': 'other', 'new': 90},
        {'foo': 'other', 'toto': [3, 4, None], 'new': 90},
    ),
    (
        {},
        {'foo': 'bar', 'toto': 'tata'},
        {'foo': 'bar', 'toto': 'tata'},
    ),
    (
        {'foo': 'bar', 'toto': 'tata'},
        {},
        {'foo': 'bar', 'toto': 'tata'},
    ),
))
def test_set_molecule_meta_molecule(prior, meta, expected):
    """
    Test for :func:`SetMoleculeMeta.run_molecule` .
    """
    molecule = Molecule()
    molecule.meta = copy.copy(prior)
    processor = SetMoleculeMeta(**meta)
    processor.run_molecule(molecule)
    assert not are_different(molecule.meta, expected)


@pytest.mark.parametrize('priors, meta, expected', (
    (
        [{}, {'foo': 'bar'}, {'toto': {'a': 0, 'b': None}}],
        {'new': 42, 'foo': None},
        [
            {'new': 42, 'foo': None},
            {'new': 42, 'foo': None},
            {'new': 42, 'foo': None, 'toto': {'a': 0, 'b': None}},
        ],
    ),
))
def test_set_molecule_meta_system(priors, meta, expected):
    """
    Test for :func:`SetMoleculeMeta.run_system` .
    """
    system = System()
    for prior in priors:
        molecule = Molecule()
        molecule.meta = copy.copy(prior)
        system.add_molecule(molecule)
    processor = SetMoleculeMeta(**meta)
    processor.run_system(system)
    for molecule, expectation in zip(system.molecules, expected):
        assert not are_different(molecule.meta, expectation)
