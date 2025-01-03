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
Unit tests for the Go contact map generator.
"""
import pytest
from vermouth.rcsu import contact_map
from vermouth.rcsu.contact_map import GenerateContactMap
import vermouth
from vermouth.tests.helper_functions import test_molecule

@pytest.mark.parametrize('resname, atomname, expected',
                         (
                                 ('TRP', 'N', 1.64),
                                 ('TRP', 'NON', 0.00),
                                 ('NON', 'N', 0.00),
                                 ('NON', 'NON', 0.00)
                         ))
def test_get_vdw_radius(resname, atomname, expected):
    result = contact_map._get_vdw_radius(resname, atomname)
    assert result == expected


@pytest.mark.parametrize('resname, atomname, expected',
                         (
                                 ('TRP', 'N', 3),
                                 ('TRP', 'NON', 0),
                                 ('NON', 'N', 0),
                                 ('NON', 'NON', 0)
                         ))
def test_get_atype(resname, atomname, expected):
    result = contact_map._get_atype(resname, atomname)
    assert result == expected