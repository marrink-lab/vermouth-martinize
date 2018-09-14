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
Unit tests for the :mod:`vermouth.processors.tune_cystein_bridges` module.
"""

# The redefined-outer-name check from pylint wrongly catches the use of pytest
# fixtures.
# pylint: disable=redefined-outer-name

import copy
import pytest
from vermouth.processors import tune_cystein_bridges
from .test_edge_tuning import simple_protein  # pylint: disable=unused-import


@pytest.fixture
def simple_protein_pruned(simple_protein):
    """
    Protein-like molecule with cystein bridges removed using
    :func:`tune_cystein_bridges.remove_cystein_bridge_edges`.
    """
    graph = copy.deepcopy(simple_protein)
    tune_cystein_bridges.remove_cystein_bridge_edges(graph)
    return graph


@pytest.mark.parametrize('edge', ((5, 13), (7, 11)))
def test_remove_cystein_bridge_edges_remove(simple_protein_pruned, edge):
    """
    Assure that the expected edges were removed by
    :func:`tune_cystein_bridges.remove_cystein_bridge_edges`.
    """
    assert edge not in simple_protein_pruned.edges


@pytest.mark.parametrize('edge', (
    (0, 1), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9),
    (8, 10), (10, 11), (10, 12), (12, 13), (12, 14), (14, 15),
    (14, 16), (16, 17), (1, 17), (3, 15),
))
def test_remove_cystein_bridge_edges_kept(simple_protein_pruned, edge):
    """
    Assure that :func:`tune_cystein_bridges.remove_cystein_bridge_edges`
    does not remove edges it should not remove.
    """
    assert edge in simple_protein_pruned.edges
