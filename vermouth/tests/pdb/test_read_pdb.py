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
Unittests for the PDB reader.
"""

import numpy as np
import pytest

from vermouth.pdb.pdb import PDBParser
from vermouth.tests.datafiles import PDB_MULTIMODEL


def single_model(with_conect):
    """
    Provide a PDB representation of a single molecule.
    """
    pdb = '''
HETATM    1  C01 UNK     1       3.939  -2.095  -4.491  0.00  0.00           C   
HETATM    2  C02 UNK     1       4.825  -2.409  -5.534  0.00  0.00           C   
HETATM    3  C03 UNK     1       5.466  -3.658  -5.560  0.00  0.00           C   
HETATM    4  C04 UNK     1       5.207  -4.601  -4.552  0.00  0.00           C   
HETATM    5  C05 UNK     1       4.320  -4.288  -3.508  0.00  0.00           C   
HETATM    6  C06 UNK     1       3.693  -3.031  -3.473  0.00  0.00           C   
HETATM    7  C07 UNK     1       2.736  -2.681  -2.319  0.00  0.00           C   
HETATM    8  C08 UNK     1       2.386  -3.959  -1.535  0.00  0.00           C   
HETATM    9  C09 UNK     1       1.429  -3.608  -0.380  0.00  0.00           C   
HETATM   10  C10 UNK     1       1.079  -4.886   0.404  0.00  0.00           C   
HETATM   11  H01 UNK     1       3.217  -1.965  -1.652  0.00  0.00           H   
HETATM   12  H02 UNK     1       1.905  -4.675  -2.201  0.00  0.00           H   
HETATM   13  H03 UNK     1       1.824  -2.243  -2.724  0.00  0.00           H   
HETATM   14  H04 UNK     1       1.910  -2.892   0.287  0.00  0.00           H   
HETATM   15  H05 UNK     1       3.298  -4.397  -1.129  0.00  0.00           H   
HETATM   16  H06 UNK     1       0.598  -5.602  -0.262  0.00  0.00           H   
HETATM   17  H07 UNK     1       3.435  -1.113  -4.469  0.00  0.00           H   
HETATM   18  H08 UNK     1       5.018  -1.673  -6.334  0.00  0.00           H   
HETATM   19  H09 UNK     1       6.172  -3.900  -6.373  0.00  0.00           H   
HETATM   20  H10 UNK     1       5.699  -5.588  -4.579  0.00  0.00           H   
HETATM   21  H11 UNK     1       4.116  -5.028  -2.715  0.00  0.00           H   
HETATM   22  H12 UNK     1       0.517  -3.170  -0.786  0.00  0.00           H   
HETATM   23  H13 UNK     1       0.401  -4.638   1.221  0.00  0.00           H   
HETATM   24  H14 UNK     1       1.991  -5.324   0.810  0.00  0.00           H   
'''
    if with_conect:
        pdb += '''
CONECT    1    2                                                                
CONECT    1    2                                                                
CONECT    1    6   17                                                           
CONECT    2    1                                                                
CONECT    2    1                                                                
CONECT    2    3   18                                                           
CONECT    3    2                                                                
CONECT    3    4                                                                
CONECT    3    4                                                                
CONECT    3   19                                                                
CONECT    4    3                                                                
CONECT    4    3                                                                
CONECT    4    5   20                                                           
CONECT    5    4                                                                
CONECT    5    6                                                                
CONECT    5    6                                                                
CONECT    5   21                                                                
CONECT    6    1                                                                
CONECT    6    5                                                                
CONECT    6    5                                                                
CONECT    6    7                                                                
CONECT    7    6    8   11   13                                                 
CONECT    8    7    9   12   15                                                 
CONECT    9    8   10   14   22                                                 
CONECT   10    9   16   23   24                                                 
CONECT   11    7                                                                
CONECT   12    8                                                                
CONECT   13    7                                                                
CONECT   14    9                                                                
CONECT   15    8                                                                
CONECT   16   10                                                                
CONECT   17    1                                                                
CONECT   18    2                                                                
CONECT   19    3                                                                
CONECT   20    4                                                                
CONECT   21    5                                                                
CONECT   22    9                                                                
CONECT   23   10                                                                
CONECT   24   10                                                                
'''
    pdb += 'END'
    return pdb


def multi_mol(with_conect):
    pdb = '''
ATOM      1  C01 UNK     1       3.939  -2.095  -4.491  0.00  0.00           C2+   
ATOM      2  C02 UNK     1       4.825  -2.409  -5.534  0.00  0.00              
TER       3      UNK     1
ATOM      4  C03 UNK     1       5.466  -3.658  -5.560  0.00  0.00           C   
ATOM      5  C04 UNK     1       5.207  -4.601  -4.552  0.00  0.00           C   
ATOM      6  C05 UNK     1       5.207  -4.601  -4.552  0.00  0.00           C   
TER       7      UNK     1
'''
    if with_conect:
        pdb += '''
CONECT    1    2                                                                
CONECT    5    4    6    
'''
    return pdb


def merged_mol(with_conect):
    pdb = '''
ATOM      1  C01 UNK     1       3.939  -2.095  -4.491  0.00  0.00           C   
ATOM      2  C02 UNK     1       4.825  -2.409  -5.534  0.00  0.00           C   
TER       3      UNK     1
ATOM      4  C03 UNK     1       5.466  -3.658  -5.560  0.00  0.00           C   
ATOM      5  C04 UNK     1       5.207  -4.601  -4.552  0.00  0.00           C   
ATOM      6  C05 UNK     1       5.207  -4.601  -4.552  0.00  0.00           C   
TER       7      UNK     1
'''
    if with_conect:
        pdb += '''
CONECT    1    2                                                                
CONECT    5    4    6    2
'''
    return pdb


@pytest.mark.parametrize('pdbstr, ignh, nnodesnedges', (
    [single_model(True), False, ((24, 24),)],
    [single_model(False), False, ((24, 0),)],
    [single_model(True), True, ((10, 10),)],
    [single_model(False), True, ((10, 0),)],
    [multi_mol(True), None, ((2, 1), (3, 2))],
    [multi_mol(False), None, ((2, 0), (3, 0))],
    [merged_mol(True), None, ((5, 4),)],
    [merged_mol(False), None, ((2, 0), (3, 0))],
))
def test_single_model(pdbstr, ignh, nnodesnedges):
    parser = PDBParser(ignh=ignh)
    mols = list(parser.parse(pdbstr.splitlines()))
    assert len(mols) == len(nnodesnedges)
    for mol, nnodes_edges in zip(mols, nnodesnedges):
        nnodes, nedges = nnodes_edges
        assert len(mol.nodes) == nnodes
        assert len(mol.edges) == nedges


@pytest.mark.parametrize('ignh', [True, False])
@pytest.mark.parametrize('modelidx', range(1, 16))
def test_integrative(ignh, modelidx):
    parser = PDBParser(ignh=ignh, modelidx=modelidx)
    with open(str(PDB_MULTIMODEL)) as pdb_file:
        mols = list(parser.parse(pdb_file))
    assert len(mols) == 3  # 3 chains
    for mol in mols:
        if ignh:
            assert len(mol.nodes) == 441
            assert len(mol.edges) == 0  # No CONECT records
        else:
            assert len(mol.nodes) == 904
            assert len(mol.edges) == 0


def test_altloc(caplog):
    pdbstr = """
ATOM   1300  CA AHIS A 194     -13.902 -22.133  70.272  0.56 53.66
ATOM   1301  CA BHIS A 194     -13.910 -22.208  70.255  0.44 53.81
"""
    parser = PDBParser()
    mols = list(parser.parse(pdbstr.splitlines()))
    assert len(mols) == 1
    mol = mols[0]
    assert len(mol.nodes) == 1
    assert any(rec.levelname == 'WARNING' for rec in caplog.records)


def test_atom_attributes():
    """
    Test that atom attributes are parsed and set correctly
    """
    pdb = multi_mol(False)
    parser = PDBParser()
    mols = list(parser.parse(pdb.splitlines()))
    assert len(mols) == 2
    nodes = [
        {
            0: {'atomname': 'C01', 'atomid': 1, 'resid': 1,
                'resname': 'UNK', 'chain':'', 'altloc': '',
                'insertion_code': '', 'occupancy': 0.0, 'temp_factor': 0.0,
                'position': np.array([0.3939,  -0.2095,  -0.4491]),
                'charge': 2.0, 'element': 'C'},
            1: {'atomname': 'C02', 'atomid': 2, 'resid': 1,
                'resname': 'UNK', 'chain':'', 'altloc': '',
                'insertion_code': '', 'occupancy': 0.0, 'temp_factor': 0.0,
                'position': np.array([0.4825,  -0.2409,  -0.5534]),
                'charge': 0, 'element': 'C'}
        },
        {
            0: {'atomname': 'C03', 'atomid': 4, 'resid': 1,
                'resname': 'UNK', 'chain': '', 'altloc': '',
                'insertion_code': '', 'occupancy': 0.0, 'temp_factor': 0.0,
                'position': np.array([0.5466,  -0.3658,  -0.5560]),
                'charge': 0, 'element': 'C'},
            1: {'atomname': 'C04', 'atomid': 5, 'resid': 1,
                'resname': 'UNK', 'chain': '', 'altloc': '',
                'insertion_code': '', 'occupancy': 0.0, 'temp_factor': 0.0,
                'position': np.array([0.5207,  -0.4601,  -0.4552]),
                'charge': 0, 'element': 'C'},
            2: {'atomname': 'C05', 'atomid': 6, 'resid': 1,
                'resname': 'UNK', 'chain': '', 'altloc': '',
                'insertion_code': '', 'occupancy': 0.0, 'temp_factor': 0.0,
                'position': np.array([0.5207, -0.4601, -0.4552]),
                'charge': 0, 'element': 'C'}
        }
    ]
    for mol, n_attrs in zip(mols, nodes):
        for n_idx in mol.nodes:
            assert set(n_attrs[n_idx].keys()) == set(mol.nodes[n_idx].keys())
            for attr in mol.nodes[n_idx]:
                assert attr in n_attrs[n_idx]
                if attr == 'position':
                    assert np.allclose(n_attrs[n_idx][attr], mol.nodes[n_idx][attr])
                else:
                    assert n_attrs[n_idx][attr] == mol.nodes[n_idx][attr]
