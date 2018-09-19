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
Unittests for the PDB writer.
"""

# Pylint is wrongly complaining about fixtures.
# pylint: disable=redefined-outer-name


import numpy as np
import networkx as nx

import pytest

import vermouth
from vermouth.molecule import Molecule
from vermouth.pdb import pdb


@pytest.fixture
def dummy_system():
    """
    Build a system with 3 dummy molecules.

    All nodes have the following attributes:

    * `atomname`
    * `resname`
    * `resid`
    * `chain` set to `''`
    * `element` set to the value of `atomname`
    * `positions`

    Some nodes have a `charge` attribute set to an integer charge. The
    molecules are connected.
    """
    nodes = (
        {'atomname': 'A', 'resname': 'A', 'resid': 1, },
        {'atomname': 'B', 'resname': 'A', 'resid': 1, 'charge': -1},
        {'atomname': 'C', 'resname': 'A', 'resid': 2, 'charge': 1},
        {'atomname': 'D', 'resname': 'A', 'resid': 3, },
        {'atomname': 'E', 'resname': 'A', 'resid': 4, },
        {'atomname': 'F', 'resname': 'B', 'resid': 4, },
        {'atomname': 'G', 'resname': 'B', 'resid': 4, },
        {'atomname': 'H', 'resname': 'B', 'resid': 4, },
    )
    edges = [(0, 1), (2, 3), (4, 5), (5, 6), (5, 7)]
    graph = nx.Graph()
    for idx, node in enumerate(nodes):
        node['chain'] = ''
        node['element'] = node['atomname']
        node['position'] = np.array([1, 2, -3])
        graph.add_node(idx, **node)
    graph.add_edges_from(edges)
    molecules = [Molecule(graph.subgraph(component))
                 for component in nx.connected_components(graph)]
    system = vermouth.system.System()
    system.molecules = molecules
    return system


@pytest.fixture
def missing_pos_system(dummy_system):
    """
    Build a system of dummy molecules with some nodes lacking a `position`
    attribute.
    """
    for node in dummy_system.molecules[1].nodes.values():
        del node['position']
    return dummy_system


def test_conect(dummy_system):
    """
    Test that the CONECT record is written as expected. Test also that the
    overall format is correct.
    """
    pdb_found = pdb.write_pdb_string(dummy_system, conect=True, omit_charges=False)
    expected = '''
ATOM      1 A    A       1      10.000  20.000 -30.000  1.00  0.00          A   
ATOM      2 B    A       1      10.000  20.000 -30.000  1.00  0.00          B 1-
TER       3      A       1 
ATOM      4 C    A       2      10.000  20.000 -30.000  1.00  0.00          C 1+
ATOM      5 D    A       3      10.000  20.000 -30.000  1.00  0.00          D   
TER       6      A       3 
ATOM      7 E    A       4      10.000  20.000 -30.000  1.00  0.00          E   
ATOM      8 F    B       4      10.000  20.000 -30.000  1.00  0.00          F   
ATOM      9 G    B       4      10.000  20.000 -30.000  1.00  0.00          G   
ATOM     10 H    B       4      10.000  20.000 -30.000  1.00  0.00          H   
TER      11      B       4 
CONECT    1    2
CONECT    4    5
CONECT    7    8
CONECT    8    9   10
END
'''
    assert pdb_found.strip() == expected.strip()


def test_write_failure_missing_pos(missing_pos_system):
    """
    Make sure the writing fails when coordinates are missing and
    `nan_missing_pos` is not set. (Shall be `False` by default.)
    """
    with pytest.raises(KeyError):
        pdb.write_pdb_string(missing_pos_system)


def test_write_success_missing_pos(missing_pos_system):
    """
    Make sure the writing succeed when coordinates are missing and
    `nan_missing_pos` is `True`.
    """
    pdb_found = pdb.write_pdb_string(
        missing_pos_system,
        conect=False,
        omit_charges=False,
        nan_missing_pos=True,  # Argument of interest!
    )
    expected = '''
ATOM      1 A    A       1      10.000  20.000 -30.000  1.00  0.00          A   
ATOM      2 B    A       1      10.000  20.000 -30.000  1.00  0.00          B 1-
TER       3      A       1 
ATOM      4 C    A       2         nan     nan     nan  1.00  0.00          C 1+
ATOM      5 D    A       3         nan     nan     nan  1.00  0.00          D   
TER       6      A       3 
ATOM      7 E    A       4      10.000  20.000 -30.000  1.00  0.00          E   
ATOM      8 F    B       4      10.000  20.000 -30.000  1.00  0.00          F   
ATOM      9 G    B       4      10.000  20.000 -30.000  1.00  0.00          G   
ATOM     10 H    B       4      10.000  20.000 -30.000  1.00  0.00          H   
TER      11      B       4 
END
'''
    assert pdb_found.strip() == expected.strip()
