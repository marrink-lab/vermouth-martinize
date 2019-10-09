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

import vermouth
from vermouth.pdb.pdb import read_pdb
from vermouth.processors import tune_cystein_bridges
from vermouth.utils import distance
from vermouth.molecule import Choice
from .test_edge_tuning import simple_protein  # pylint: disable=unused-import
from .datafiles import PDB_CYS


@pytest.fixture
def cys_protein():
    """
    Read a PDB file describing a protein with a cystein bridge and cystein
    residues that are not involved in a bridge. Embed the content of the PDB in
    a :class:`vermouth.system.System`.

    Note that the PDB file does contain water, ions, and ligands in addition to
    the protein. The ligand bonds are explicitely provided as CONECT records.
    The edges from the CONECT records are removed from the system, so there is
    no edge left.
    """
    molecules = read_pdb(PDB_CYS)
    assert len(molecules) == 1
    molecule = molecules[0]
    molecule.remove_edges_from(list(molecule.edges))
    system = vermouth.system.System()
    system.molecules = [molecule]
    assert len(system.molecules[0].edges) == 0
    return system


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


def test_add_cystein_bridges_threshold(cys_protein):
    """
    Test that :class:`vermouth.AddCysteinBridgesThreshold` detects a cystein
    bridge in a PDB file.
    """
    processor = vermouth.AddCysteinBridgesThreshold(threshold=0.22)
    system = processor.run_system(cys_protein)
    for edge in system.molecules[0].edges:
        node_a = system.molecules[0].nodes[edge[0]]
        node_b = system.molecules[0].nodes[edge[1]]
        print('*', node_a, '--', node_b)
        print(distance(node_a['position'], node_b['position']))
    assert len(system.molecules[0].edges) == 1
    # According to SSBOND record, cysbond should be between resid 171 and 876.
    # Let's find the associated atoms.
    atoms = tuple(system.molecules[0].find_atoms(resname='CYS', atomname='SG',
                                                 resid=Choice([171, 876])))
    assert atoms in system.molecules[0].edges


def test_remove_cystein_bridge_edges_processor(cys_protein):
    """
    Test that :class:`vermouth.RemoveCysteinBridgeEdges` removes edges.
    """
    print(cys_protein.molecules[0].nodes[1285], cys_protein.molecules[0].nodes[3508])
    cys_protein.molecules[0].add_edge(1285, 3508)
    processor = vermouth.RemoveCysteinBridgeEdges()
    processor.run_system(cys_protein)
    # According to SSBOND record, cysbond should be between resid 171 and 876.
    # Let's find the associated atoms.
    atoms = tuple(cys_protein.molecules[0].find_atoms(resname='CYS',
                                                      atomname='SG',
                                                      resid=Choice([171, 876])))
    assert atoms not in cys_protein.molecules[0].edges
