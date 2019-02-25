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
Test the functions required to use DSSP.
"""

import os
import itertools

import pytest

import vermouth
from vermouth.dssp import dssp
from vermouth.pdb.pdb import read_pdb
from vermouth.tests.datafiles import (
    PDB_PROTEIN,
    DSSP_OUTPUT,
)

DSSP_EXECUTABLE = os.environ.get('VERMOUTH_TEST_DSSP', 'dssp')
SECSTRUCT_1BTA = list('CEEEEETTTCCSHHHHHHHHHHHHTCCTTCCCSHHHHHHHHTTT'
                      'SCSSEEEEEESTTHHHHTTTSSHHHHHHHHHHHHHTTCCEEEEEC')


# TODO: The code is very repetitive. There may be a way to refactor it with
# clever use of parametrize and fixtures.
class TestAnnotateResidues:
    """
    Tests for the :class:`dssp.AnnotateResidues` processor.
    """
    @staticmethod
    def build_molecule(nresidues):
        """
        Build a dummy molecule with the requested number of residues and 3
        atoms per residue.
        """
        molecule = vermouth.molecule.Molecule()
        residue_template = vermouth.molecule.Molecule()
        residue_template.add_nodes_from(
            (idx, {'chain':'', 'atomname':str(idx), 'resname': 'DUMMY', 'resid': 1})
            for idx in range(3)
        )
        for _ in range(nresidues):
            molecule.merge_molecule(residue_template)
        return molecule

    @staticmethod
    def sequence_from_mol(molecule, attribute, default=None):
        """
        Extract the content of an attribute for each node of a molecule.
        """
        return [node.get(attribute, default) for node in molecule.nodes.values()]

    def sequence_from_system(self, system, attribute, default=None):
        """
        Extract the content of an attribute for each node of a system.
        """
        return list(itertools.chain(*(
            self.sequence_from_mol(molecule, attribute, default)
            for molecule in system.molecules
        )))

    @pytest.mark.parametrize('nres', (0, 1, 3, 10))
    def test_build_molecule(self, nres):
        """
        :meth:`build_molecule` and :meth:`sequence_from_mol` work as excpected.
        """
        expected_resid = list(itertools.chain(
            *([idx + 1] * 3 for idx in range(nres))
        ))
        expected_atomname = ['0', '1', '2'] * nres
        expected_chain = [''] * (nres * 3)
        expected_resname = ['DUMMY'] * (nres * 3)
        molecule = self.build_molecule(nres)
        assert self.sequence_from_mol(molecule, 'resid') == expected_resid
        assert self.sequence_from_mol(molecule, 'resname') == expected_resname
        assert self.sequence_from_mol(molecule, 'chain') == expected_chain
        assert self.sequence_from_mol(molecule, 'atomname') == expected_atomname

    @pytest.fixture
    def single_mol_system(self):
        """
        Build a system with a single molecule that count 5 residues of 3 atoms
        each.
        """
        molecule = self.build_molecule(5)
        system = vermouth.system.System()
        system.molecules = [molecule]
        return system

    @pytest.fixture
    def multi_mol_system_irregular(self):
        """
        Build a system with 3 molecules having 4, 5, and 6 residues,
        respectively.
        """
        system = vermouth.system.System()
        system.molecules = [self.build_molecule(nres) for nres in (4, 5, 6)]
        return system

    @pytest.fixture
    def multi_mol_system_regular(self):
        """
        Build a system with 3 molecules having each 4 residues.
        """
        system = vermouth.system.System()
        system.molecules = [self.build_molecule(4) for _ in range(3)]
        return system

    @pytest.mark.parametrize('sequence', (
        'ABCDE',
        ['A', 'B', 'C', 'D', 'E'],
        range(5),
    ))
    def test_single_molecule(self, single_mol_system, sequence):
        """
        The simple case with a single molecule and a sequence of the right size
        works as expected.
        """
        expected = list(itertools.chain(
            *([element] * 3 for element in sequence)
        ))
        processor = dssp.AnnotateResidues('test', sequence)
        processor.run_system(single_mol_system)
        found = self.sequence_from_system(single_mol_system, 'test')
        assert found == expected

    @pytest.mark.parametrize('sequence', (
        'ABCDEFGHIJKLMNO',
        list('ABCDEFGHIJKLMNO'),
        range(15),
    ))
    def test_multi_molecules_diff_sizes(self, multi_mol_system_irregular, sequence):
        """
        The case of many protein of various sizes and a sequence of the right
        size works as expected.
        """
        expected = list(itertools.chain(
            *([element] * 3 for element in sequence)
        ))
        processor = dssp.AnnotateResidues('test', sequence)
        processor.run_system(multi_mol_system_irregular)
        found = self.sequence_from_system(multi_mol_system_irregular, 'test')
        assert found == expected

    @pytest.mark.parametrize('sequence', (
        'ABCD',
        ['A', 'B', 'C', 'D'],
        range(4),
    ))
    def test_multi_molecules_cycle(self, multi_mol_system_regular, sequence):
        """
        The case with multiple molecules with all the same size and one
        sequence to repeat for each molecule works as expected.
        """
        expected = list(itertools.chain(
            *([element] * 3 for element in sequence)
        ))
        expected = expected * 3
        processor = dssp.AnnotateResidues('test', sequence)
        processor.run_system(multi_mol_system_regular)
        found = self.sequence_from_system(multi_mol_system_regular, 'test')
        assert found == expected

    def test_single_molecules_cycle_one(self, single_mol_system):
        """
        One molecule and a one element sequence to repeat over all residues of
        the molecule.
        """
        sequence = 'A'
        expected = [sequence] * (5 * 3)
        processor = dssp.AnnotateResidues('test', sequence)
        processor.run_system(single_mol_system)
        found = self.sequence_from_system(single_mol_system, 'test')
        assert found == expected


    def test_multi_molecules_cycle_one(self, multi_mol_system_irregular):
        """
        Many molecules and a one element sequence to repeat.
        """
        sequence = 'A'
        expected = [sequence] * (15 * 3)
        processor = dssp.AnnotateResidues('test', sequence)
        processor.run_system(multi_mol_system_irregular)
        found = self.sequence_from_system(multi_mol_system_irregular, 'test')
        assert found == expected

    @staticmethod
    @pytest.mark.parametrize('sequence', (
        'ABC',  # Too short
        'ABCD',  # Too short, match the length of the first molecule
        'ABCDEFGHIFKLMNOPQRSTU',  # Too long
        '',  # Empty
    ))
    def test_wrong_length(multi_mol_system_irregular, sequence):
        """
        Many molecule and a sequence that has the wrong length raises an error.
        """
        processor = dssp.AnnotateResidues('test', sequence)
        with pytest.raises(ValueError):
            processor.run_system(multi_mol_system_irregular)

    @staticmethod
    @pytest.mark.parametrize('sequence', (
        'ABC',  # Too short
        'ABCD',  # Too short, match the length of the first molecule
        'ABCDEFGHIFKLMNOPQRSTU',  # Too long
        '',  # Empty
        'ABCDEFGHIJKLMNO',  # Length of all the molecules, without filter
    ))
    def test_wrong_length_with_filter(multi_mol_system_irregular, sequence):
        """
        Many molecules and a sequence that has the wrong length because of a
        molecule selector.
        """
        # We exclude the second molecule. The filter excludes it based on the
        # number of nodes, which is 15 because it has 5 residues with 3 nodes
        # each.
        processor = dssp.AnnotateResidues(
            'test', sequence,
            molecule_selector=lambda mol: len(mol.nodes) != (5 * 3),
        )
        with pytest.raises(ValueError):
            processor.run_system(multi_mol_system_irregular)

    @staticmethod
    def test_empty_system_empty_sequence():
        """
        There are no molecules, but the sequence is empty.
        """
        system = vermouth.system.System()
        sequence = ''
        processor = dssp.AnnotateResidues('test', sequence)
        try:
            processor.run_system(system)
        except ValueError:
            pytest.fail('Should not have raised a ValueError.')

    @staticmethod
    def test_empty_system_error():
        """
        There are no molecules, but there is a sequence. Should raise an error.
        """
        system = vermouth.system.System()
        sequence = 'not empty'
        processor = dssp.AnnotateResidues('test', sequence)
        with pytest.raises(ValueError):
            processor.run_system(system)

    @staticmethod
    def test_empty_with_filter(multi_mol_system_irregular):
        """
        There is a sequence, but no molecule are accepted by the molecule
        selector. Should raise an error.
        """
        sequence = 'not empty'
        processor = dssp.AnnotateResidues(
            'test', sequence, molecule_selector=lambda mol: False
        )
        with pytest.raises(ValueError):
            processor.run_system(multi_mol_system_irregular)

    def test_run_molecule(self, single_mol_system):
        """
        The `run_molecule` method works.
        """
        sequence = 'ABCDE'
        expected = list(itertools.chain(
            *([element] * 3 for element in sequence)
        ))
        processor = dssp.AnnotateResidues('test', sequence)
        processor.run_molecule(single_mol_system.molecules[0])
        found = self.sequence_from_system(single_mol_system, 'test')
        assert found == expected

    def test_run_molecule_not_selected(self, single_mol_system):
        """
        The molecule selector works with `run_molecule`.
        """
        sequence = 'ABCDE'
        processor = dssp.AnnotateResidues(
            'test', sequence, molecule_selector=lambda mol: False
        )
        processor.run_molecule(single_mol_system.molecules[0])
        found = self.sequence_from_system(single_mol_system, 'test')
        assert vermouth.utils.are_all_equal(found)
        assert found[0] is None


def test_read_dssp2():
    """
    Test that :func:`vermouth.dssp.dssp.read_dssp2` returns the expected
    secondary structure sequence.
    """
    with open(str(DSSP_OUTPUT)) as infile:
        secondary_structure = dssp.read_dssp2(infile)
    assert secondary_structure == SECSTRUCT_1BTA


@pytest.mark.parametrize('savefile', [True, False])
def test_run_dssp(savefile, tmpdir):
    """
    Test that :func:`vermouth.molecule.dssp.dssp.run_dssp` runs as expected and
    generate a save file only if requested.
    """
    # The test runs twice, once with the savefile set to True so we test with
    # savinf the DSSP output to file, and once with savefile set t False so we
    # do not generate the file. The "savefile" argument is set by
    # pytest.mark.parametrize.
    # The "tmpdir" argument is set by pytest and is the path to a temporary
    # directory that exists only for one iteration of the test.
    if savefile:
        path = tmpdir.join('dssp_output')
    else:
        path = None
    system = vermouth.System()
    system.add_molecule(read_pdb(str(PDB_PROTEIN)))
    secondary_structure = dssp.run_dssp(system,
                                        executable=DSSP_EXECUTABLE,
                                        savefile=path)

    # Make sure we produced the expected sequence of secondary structures
    assert secondary_structure == SECSTRUCT_1BTA

    # If we test with savefile, then we need to make sure the file is created
    # and its content corresponds to the reference (excluding the first lines
    # that are variable or contain non-essencial data read from the PDB file).
    # If we test without savefile, then we need to make sure the file is not
    # created.
    if savefile:
        assert path.exists()
        with open(str(path)) as genfile, open(str(DSSP_OUTPUT)) as reffile:
            # DSSP 3 is outputs mostly the same thing as DSSP2, though there
            # are some differences in non significant whitespaces, and an extra
            # field header. We need to normalize these differences to be able
            # to compare.
            gen = '\n'.join([
                line.strip().replace('            CHAIN', '')
                for line in genfile.readlines()[6:]
            ])
            ref = '\n'.join([line.strip() for line in reffile.readlines()[6:]])
            assert gen == ref
    else:
        # Is the directory empty?
        assert not os.listdir(str(tmpdir))
