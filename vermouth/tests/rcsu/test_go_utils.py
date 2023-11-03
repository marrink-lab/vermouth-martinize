# -*- coding: utf-8 -*-
# Copyright 2022 University of Groningen
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
Unit tests for the Go contact map reader.
"""
import pytest
import hypothesis
import hypothesis.strategies as st
from vermouth.rcsu.go_utils import _get_bead_size, _in_resid_region, get_go_type_from_attributes
from vermouth.tests.molecule_strategies import random_molecule

@pytest.mark.parametrize('atype, size', (
        ("SN4a", "small"),
        ("P3", "regular"),
        ("Tq4d", "tiny")
    ))
def test_get_bead_size(atype, size):
    assert size == _get_bead_size(atype)


@pytest.mark.parametrize('regions, resid, result', (
        # single region true
        ([(101, 120)], 115, True),
        # two regions true
        ([(101, 120), (180, 210)], 115, True),
        # two regions true; but inaccurate sorting
        ([(120, 101), (180, 210)], 115, True),
        # single region false
        ([(101, 120)], 15, False),
        # two regions false
        ([(101, 120), (180, 210)], 15, False),
    ))
def test_in_region(regions, resid, result):
    assert result == _in_resid_region(resid, regions)


@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
@hypothesis.given(random_molecule())
def test_get_go_type_from_attributes(mol):
    vs_node = len(mol.nodes)
    mol.add_node(vs_node, atype="prefix_0", chain="A", resid=5)
    found_atype = next(get_go_type_from_attributes(mol, prefix="prefix", chain="A", resid=5))
    assert found_atype == "prefix_0"

@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
@hypothesis.given(random_molecule())
def test_error_get_go_type_from_attributes(mol):
    vs_node = len(mol.nodes)
    with pytest.raises(KeyError):
        next(get_go_type_from_attributes(mol, prefix="prefix", chain="A", resid=5))
