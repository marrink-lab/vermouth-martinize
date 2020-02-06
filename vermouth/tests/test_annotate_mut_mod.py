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
Contains unittests for vermouth.processors.annotate_mut_mod.
"""

import pytest
import vermouth


@pytest.mark.parametrize('spec,expected', [
    ('', {}),
    ('-', {'chain': ''}),
    ('#', {}),
    ('-#', {'chain': ''}),
    ('A-ALA1', {'chain': 'A', 'resname': 'ALA', 'resid': 1}),
    ('A-ALA#1', {'chain': 'A', 'resname': 'ALA', 'resid': 1}),
    ('ALA1', {'resname': 'ALA', 'resid': 1}),
    ('A-ALA', {'chain': 'A', 'resname': 'ALA'}),
    ('ALA', {'resname': 'ALA'}),
    ('2', {'resid': 2}),
    ('#2', {'resid': 2}),
    ('PO4#3', {'resname': 'PO4', 'resid': 3}),
    ('PO43', {'resname': 'PO', 'resid': 43}),


])
def test_parse_residue_spec(spec, expected):
    found = vermouth.processors.annotate_mut_mod.parse_residue_spec(spec)
    assert found == expected


@pytest.mark.parametrize('dict1,dict2,expected', [
    ({}, {}, True),
    ({1: 1}, {}, False),
    ({}, {1: 1}, True),
    ({1: 1}, {1: 1}, True),
    ({1: 1}, {1: 1, 2: 2}, True),
    ({1: 1, 2: 2}, {1: 1}, False),
    ({1: 1}, {1: 3, 2: 2}, False),
    ({1: 1, 2: 2}, {1: 1, 2: 3}, False),
])
def test_subdict(dict1, dict2, expected):
    found = vermouth.processors.annotate_mut_mod._subdict(dict1, dict2)
    assert found == expected

