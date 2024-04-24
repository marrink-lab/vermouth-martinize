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
Contains unittests for vermouth.processors.merge_chains.
"""

import networkx as nx
import pytest
from vermouth.system import System
from vermouth.molecule import Molecule
from vermouth.forcefield import ForceField
from vermouth.processors.merge_chains import (
    MergeChains
)
from vermouth.tests.datafiles import (
    FF_UNIVERSAL_TEST,
)

@pytest.mark.parametrize('node_data, edge_data, merger, expected', [
    (
            [
                {'chain': 'A', 'resname': 'ALA', 'resid': 1},
                {'chain': 'A', 'resname': 'ALA', 'resid': 2},
                {'chain': 'A', 'resname': 'ALA', 'resid': 3},
                {'chain': 'B', 'resname': 'ALA', 'resid': 1},
                {'chain': 'B', 'resname': 'ALA', 'resid': 2},
                {'chain': 'B', 'resname': 'ALA', 'resid': 3}
            ],
            [(0, 1), (1, 2), (3, 4), (4, 5)],
            {"chains": ["A", "B"], "all_chains": False},
            False
    ),
    (
            [
                {'chain': 'A', 'resname': 'ALA', 'resid': 1},
                {'chain': 'A', 'resname': 'ALA', 'resid': 2},
                {'chain': 'A', 'resname': 'ALA', 'resid': 3},
                {'chain': 'B', 'resname': 'ALA', 'resid': 1},
                {'chain': 'B', 'resname': 'ALA', 'resid': 2},
                {'chain': 'B', 'resname': 'ALA', 'resid': 3}
            ],
            [(0, 1), (1, 2), (3, 4), (4, 5)],
            {"chains": [], "all_chains": True},
            False
    ),
    (
            [
                {'chain': 'A', 'resname': 'ALA', 'resid': 1},
                {'chain': 'A', 'resname': 'ALA', 'resid': 2},
                {'chain': 'A', 'resname': 'ALA', 'resid': 3},
                {'chain': None, 'resname': 'ALA', 'resid': 1},
                {'chain': None, 'resname': 'ALA', 'resid': 2},
                {'chain': None, 'resname': 'ALA', 'resid': 3}
            ],
            [(0, 1), (1, 2), (3, 4), (4, 5)],
            {"chains": [], "all_chains": True},
            True
    ),

])
def test_merge(caplog, node_data, edge_data, merger, expected):
    """
    Tests that the merging works as expected.
    """
    system = System(force_field=ForceField(FF_UNIVERSAL_TEST))
    mol = Molecule(force_field=system.force_field)
    mol.add_nodes_from(enumerate(node_data))
    mol.add_edges_from(edge_data)

    mols = nx.connected_components(mol)
    for nodes in mols:
        system.add_molecule(mol.subgraph(nodes))

    processor = MergeChains(**merger)
    caplog.clear()
    processor.run_system(system)

    if expected:
        assert any(rec.levelname == 'WARNING' for rec in caplog.records)
    else:
        assert caplog.records == []

def test_too_many_args():
    """
    Tests that error is raised when too many arguments are given.
    """
    node_data = [
                {'chain': 'A', 'resname': 'ALA', 'resid': 1},
                {'chain': 'A', 'resname': 'ALA', 'resid': 2},
                {'chain': 'A', 'resname': 'ALA', 'resid': 3},
                {'chain': 'B', 'resname': 'ALA', 'resid': 1},
                {'chain': 'B', 'resname': 'ALA', 'resid': 2},
                {'chain': 'B', 'resname': 'ALA', 'resid': 3}
                ]
    edge_data = [(0, 1), (1, 2), (3, 4), (4, 5)]

    system = System(force_field=ForceField(FF_UNIVERSAL_TEST))
    mol = Molecule(force_field=system.force_field)
    mol.add_nodes_from(enumerate(node_data))
    mol.add_edges_from(edge_data)

    mols = nx.connected_components(mol)
    for nodes in mols:
        system.add_molecule(mol.subgraph(nodes))

    merger = {"chains": ["A", "B"], "all_chains": True}

    processor = MergeChains(**merger)

    with pytest.raises(ValueError):
        processor.run_system(system)

