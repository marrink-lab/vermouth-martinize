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
Test reading a PDB file.
"""

import textwrap
import pytest

import numpy as np

from vermouth import Molecule
from vermouth.pdb import read_pdb

from vermouth.utils import are_different


@pytest.mark.parametrize('lines, model, nodes, edges', (
    (  # Empty file
        """
        """,
        0,
        (),
        (),
    ),
    (  # Only remarks
        """
        HEADER    CHAPERONE                               10-AUG-07   2QWO              
        TITLE     CRYSTAL STRUCTURE OF DISULFIDE-BOND-CROSSLINKED COMPLEX OF BOVINE     
        TITLE    2 HSC70 (1-394AA)R171C AND BOVINE AUXILIN (810-910AA)D876C IN THE      
        TITLE    3 ADP*PI FORM #1                                                       
        REMARK   3    REFINEMENT TARGET : MAXIMUM LIKELIHOOD                            
        """,
        0,
        (),
        (),
    ),
    (  # Simple case with one model and only atoms
        """
        HETATM    1  N   GLY A   4       7.291   3.975 -50.874  1.00 36.86           N  
        ATOM      2  CA  GLY A   4       7.025   3.490 -49.481  1.00 35.72           C  
        HETATM    3  C   GLY A   5       8.158   3.805 -48.514  1.00 35.39           C  
        ATOM      4  O   GLY A   5       9.334   3.581 -48.839  1.00 36.00           O  
        """,
        0,
        (
            (0, {
                'record_type': 'HETATM',
                'atomid': 1,
                'atomname': 'N',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 4,
                'insertion_code': '',
                'position': np.array([0.7291, 0.3975, -5.0874]),
                'occupancy': 1.00,
                'temp_factor': 36.86,
                'element': 'N',
                'charge': None,
            }),
            (1, {
                'record_type': 'ATOM',
                'atomid': 2,
                'atomname': 'CA',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 4,
                'insertion_code': '',
                'position': np.array([0.7025, 0.3490, -4.9481]),
                'occupancy': 1.00,
                'temp_factor': 35.72,
                'element': 'C',
                'charge': None,
            }),
            (2, {
                'record_type': 'HETATM',
                'atomid': 3,
                'atomname': 'C',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 5,
                'insertion_code': '',
                'position': np.array([0.8158, 0.3805, -4.8514]),
                'occupancy': 1.00,
                'temp_factor': 35.39,
                'element': 'C',
                'charge': None,
            }),
            (3, {
                'record_type': 'ATOM',
                'atomid': 4,
                'atomname': 'O',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 5,
                'insertion_code': '',
                'position': np.array([0.9334, 0.3581, -4.8839]),
                'occupancy': 1.00,
                'temp_factor': 36.00,
                'element': 'O',
                'charge': None,
            }),
        ),
        (),
    ),
    (  # Simple case with one model and only atoms and connect records
        """
        ATOM      1  N   GLY A   4       7.291   3.975 -50.874  1.00 36.86           N+1
        ATOM      2  CA  GLY A   4       7.025   3.490 -49.481  1.00 35.72           C  
        TER       3
        ATOM      4  C   GLY B   5       8.158   3.805 -48.514  1.00 35.39           C  
        ATOM      5  O   GLY B   5       9.334   3.581 -48.839  1.00 36.00           O-1
        TER       6
        CONECT    1    2    4                                                           
        CONECT    2    4                                                                
        """,
        0,
        (
            (0, {
                'record_type': 'ATOM',
                'atomid': 1,
                'atomname': 'N',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 4,
                'insertion_code': '',
                'position': np.array([0.7291, 0.3975, -5.0874]),
                'occupancy': 1.00,
                'temp_factor': 36.86,
                'element': 'N',
                'charge': 1,
            }),
            (1, {
                'record_type': 'ATOM',
                'atomid': 2,
                'atomname': 'CA',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 4,
                'insertion_code': '',
                'position': np.array([0.7025, 0.3490, -4.9481]),
                'occupancy': 1.00,
                'temp_factor': 35.72,
                'element': 'C',
                'charge': None,
            }),
            (2, {
                'record_type': 'ATOM',
                'atomid': 4,
                'atomname': 'C',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'B',
                'resid': 5,
                'insertion_code': '',
                'position': np.array([0.8158, 0.3805, -4.8514]),
                'occupancy': 1.00,
                'temp_factor': 35.39,
                'element': 'C',
                'charge': None,
            }),
            (3, {
                'record_type': 'ATOM',
                'atomid': 5,
                'atomname': 'O',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'B',
                'resid': 5,
                'insertion_code': '',
                'position': np.array([0.9334, 0.3581, -4.8839]),
                'occupancy': 1.00,
                'temp_factor': 36.00,
                'element': 'O',
                'charge': -1,
            }),
        ),
        ((0, 1), (0, 2), (1, 2)),
    ),
    (  # Two models, get the first one
        """
        MODEL
        ATOM      1  N   GLY A   4       7.291   3.975 -50.874  1.00 36.86           N  
        ATOM      2  CA  GLY A   4       7.025   3.490 -49.481  1.00 35.72           C  
        ENDMDL
        MODEL
        ATOM      3  C   GLY A   5       8.158   3.805 -48.514  1.00 35.39           C  
        ATOM      4  O   GLY A   5       9.334   3.581 -48.839  1.00 36.00           O  
        ENDMDL
        CONECT    1    2                                                                
        """,
        0,
        (
            (0, {
                'record_type': 'ATOM',
                'atomid': 1,
                'atomname': 'N',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 4,
                'insertion_code': '',
                'position': np.array([0.7291, 0.3975, -5.0874]),
                'occupancy': 1.00,
                'temp_factor': 36.86,
                'element': 'N',
                'charge': None,
            }),
            (1, {
                'record_type': 'ATOM',
                'atomid': 2,
                'atomname': 'CA',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 4,
                'insertion_code': '',
                'position': np.array([0.7025, 0.3490, -4.9481]),
                'occupancy': 1.00,
                'temp_factor': 35.72,
                'element': 'C',
                'charge': None,
            }),
        ),
        ((0, 1), ),
    ),
    (  # Two models, get the second one
        """
        MODEL
        ATOM      1  N   GLY A   4       7.291   3.975 -50.874  1.00 36.86           N  
        ATOM      2  CA  GLY A   4       7.025   3.490 -49.481  1.00 35.72           C  
        ENDMDL
        MODEL
        ATOM      1  C   GLY A   5       8.158   3.805 -48.514  1.00 35.39           C  
        ATOM      2  O   GLY A   5       9.334   3.581 -48.839  1.00 36.00           O  
        ENDMDL
        CONECT    1    2                                                                
        """,
        1,
        (
            (0, {
                'record_type': 'ATOM',
                'atomid': 1,
                'atomname': 'C',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 5,
                'insertion_code': '',
                'position': np.array([0.8158, 0.3805, -4.8514]),
                'occupancy': 1.00,
                'temp_factor': 35.39,
                'element': 'C',
                'charge': None,
            }),
            (1, {
                'record_type': 'ATOM',
                'atomid': 2,
                'atomname': 'O',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 5,
                'insertion_code': '',
                'position': np.array([0.9334, 0.3581, -4.8839]),
                'occupancy': 1.00,
                'temp_factor': 36.00,
                'element': 'O',
                'charge': None,
            }),
        ),
        ((0, 1), ),
    ),
    (  # Two models with model ID, get the second one
        """
        MODEL         1
        ATOM      1  N   GLY A   4       7.291   3.975 -50.874  1.00 36.86           N  
        ATOM      2  CA  GLY A   4       7.025   3.490 -49.481  1.00 35.72           C  
        ENDMDL
        MODEL         2
        ATOM      1  C   GLY A   5       8.158   3.805 -48.514  1.00 35.39           C  
        ATOM      2  O   GLY A   5       9.334   3.581 -48.839  1.00 36.00           O  
        ENDMDL
        CONECT    1    2                                                                
        """,
        1,
        (
            (0, {
                'record_type': 'ATOM',
                'atomid': 1,
                'atomname': 'C',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 5,
                'insertion_code': '',
                'position': np.array([0.8158, 0.3805, -4.8514]),
                'occupancy': 1.00,
                'temp_factor': 35.39,
                'element': 'C',
                'charge': None,
            }),
            (1, {
                'record_type': 'ATOM',
                'atomid': 2,
                'atomname': 'O',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 5,
                'insertion_code': '',
                'position': np.array([0.9334, 0.3581, -4.8839]),
                'occupancy': 1.00,
                'temp_factor': 36.00,
                'element': 'O',
                'charge': None,
            }),
        ),
        ((0, 1), ),
    ),

    (  # No explicit element
        """
        ATOM      2  CA  GLY A   4       7.025   3.490 -49.481  1.00 35.72              
        """,
        0,
        (
            (0, {
                'record_type': 'ATOM',
                'atomid': 2,
                'atomname': 'CA',
                'altloc': '',
                'resname': 'GLY',
                'chain': 'A',
                'resid': 4,
                'insertion_code': '',
                'position': np.array([0.7025, 0.3490, -4.9481]),
                'occupancy': 1.00,
                'temp_factor': 35.72,
                'element': 'C',
                'charge': None,
            }),
        ),
        (),
    ),
))
def test_read_pdb(tmpdir, lines, model, nodes, edges):
    pdb_path = str(tmpdir / 'test.pdb')
    with open(pdb_path, 'w') as outfile:
        outfile.write(textwrap.dedent(lines))

    read_molecule = read_pdb(pdb_path, model=model)
    expected_molecule = Molecule()
    expected_molecule.add_nodes_from(nodes)
    expected_molecule.add_edges_from(edges)

    assert read_molecule == expected_molecule
