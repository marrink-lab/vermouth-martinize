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
Test that force field files are properly read.
"""

import textwrap
import pytest
import numpy as np
import vermouth.ffinput
import vermouth.forcefield
import vermouth.molecule

class TestApplyLinks:
    @staticmethod
    @pytest.mark.parametrize('links, interactions, edges, idx, inttype',
        (("""
         [ link ]
         [ bonds ]
         BB +BB  1  0.350  1250
         """,
         [vermouth.molecule.Interaction(
               atoms=(0, 1), parameters=['1','0.350','1250'], meta={}),
         ],
         1,
         [1,2],
         'bonds'),
        ("""
         [ link ]
         [ bonds ]
         BB   SC1   1  0.350  1250
         """,
         [vermouth.molecule.Interaction(
             atoms=(2, 3), parameters=['1','0.350','1250'], meta={})],
         1,
         [3,3],
         'bonds'),
        ("""
         [ link ]
         [ angles ]
         BB  +BB  +SC1  1  125  250
         """,
         [vermouth.molecule.Interaction(
              atoms=(1, 2, 3), parameters=['1','125','250'], meta={})],
         2,
         [2,3],
         'angles')
        ))

    def test_add_interaction_and_edge(links, interactions, edges, idx, inttype):
        lines = """
        [ moleculetype ]
        GLY  1
        [ atoms ]
        ;id  type resnr residu atom cgnr   charge
         1   SP2   1     GLY    BB     1      0
        [ moleculetype ]
        ALA  1
        [ atoms ]
        ;id  type resnr residu atom cgnr   charge
         1   SP2   1     ALA    BB     1      0
         2   SC2   1     ALA    SC1     1      0
        """
        lines = lines + links
        lines = textwrap.dedent(lines).splitlines()
        FF = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, FF)

        new_mol = FF.blocks['GLY'].to_molecule()
        new_mol.merge_molecule(FF.blocks['GLY'])
        new_mol.merge_molecule(FF.blocks['ALA'])

        new_mol.apply_link_between_residues(FF.links[0], resids=[idx[0], idx[1]])
        print(new_mol.interactions[inttype])
        assert new_mol.interactions[inttype] == interactions
        assert len(new_mol.edges) == edges

    @staticmethod
    @pytest.mark.parametrize('links, idx',
        (
	(# no match considering the order parameter
         """
         [ link ]
         [ bonds ]
         BB   +BB  1  0.350  1250""",
         [1,4],
         ),
        (# no match due to incorrect atom name
         """
         [ link ]
         [ bonds ]
         BB   SC5   1  0.350  1250
         """,
         [3,3])))
    def test_link_failure(links, idx):
        lines = """
        [ moleculetype ]
        GLY  1
        [ atoms ]
        ;id  type resnr residu atom cgnr   charge
         1   SP2   1     GLY    BB     1      0
        [ moleculetype ]
        ALA  1
        [ atoms ]
        ;id  type resnr residu atom cgnr   charge
         1   SP2   1     ALA    BB     1      0
         2   SC2   1     ALA    SC1    1      0
        """
        lines = lines + links
        lines = textwrap.dedent(lines).splitlines()
        FF = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, FF)

        new_mol = FF.blocks['GLY'].to_molecule()
        new_mol.merge_molecule(FF.blocks['GLY'])
        new_mol.merge_molecule(FF.blocks['ALA'])

        with pytest.raises(KeyError):
              new_mol.apply_link_between_residues(FF.links[0], resids=[idx[0], idx[1]])

