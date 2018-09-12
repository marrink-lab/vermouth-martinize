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


import numpy as np
import networkx as nx
import vermouth
from vermouth.molecule import Molecule
from vermouth.pdb import pdb


def test_conect():
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
        graph.add_node(idx, **node, position=np.array([1, 2, -3]))
    graph.add_edges_from(edges)
    molecules = [Molecule(graph.subgraph(component))
                 for component in nx.connected_components(graph)]
    system = vermouth.system.System()
    system.molecules = molecules
    pdb_found = pdb.write_pdb_string(system, conect=True, omit_charges=False)
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
CONECT    2    1
CONECT    4    5
CONECT    5    4
CONECT    7    8
CONECT    8    7    9   10
CONECT    9    8
CONECT   10    8
END
'''
    assert pdb_found.strip() == expected.strip()
