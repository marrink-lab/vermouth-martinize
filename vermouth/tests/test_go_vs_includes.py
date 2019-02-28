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
Test the :class:`vermouth.processors.go_vs_includes.GoVirtIncludes` processor.
"""

# Pylint issues false warnings because of pytest's fixtures.
# pylint: disable=redefined-outer-name

import pytest
from vermouth import Molecule
from vermouth.processors import GoVirtIncludes

@pytest.fixture(params=(0, 4))
def molecule_for_go(request):
    """
    Molecule to test virtual site-based GoMartini.

    The molecule has a variable number of residue set in the fixture parameter.
    Each residue has a backbone and a side chain beads. It also has some post
    sections lines that should not disapear when running the processor, and a
    moltype name as this is required.
    """
    n_residues = request.param
    molecule = Molecule()
    for residue_idx in range(n_residues):
        base_atom_idx = residue_idx * 2
        for relative_atom_idx, atomname in enumerate(('BB', 'SC1')):
            molecule.add_node(
                base_atom_idx + relative_atom_idx,
                resid=residue_idx,
                resname='XX',
                atomname=atomname,
                charge_group=base_atom_idx,
                chain='A',
                # Should be an array, but it is not relevant for the test
                position=[0, 0, 0],
            )
    molecule.meta['post_section_lines'] = {
        'exclusions': ['I exist'],
        'other': ['Hello'],
    }
    molecule.meta['moltype'] = 'TEST'
    return molecule


def test_go_virt_includes(molecule_for_go):
    """
    Test the processor in normal conditions.
    """
    molecule = molecule_for_go
    initial_n_backbone = len(molecule.nodes) // 2  # Each residues has 2 atoms
    processor = GoVirtIncludes()
    processor.run_molecule(molecule)

    expected_post = {
        'exclusions': ['I exist', '#include "TEST_exclusions_VirtGoSites.itp"'],
        'other': ['Hello'],
    }

    assert molecule.meta['post_section_lines'] == expected_post
    # Each residue in the initial molecule has a backbone and a side chain
    # beads. We create one virtual site per backbone beads, so we expect to
    # have a side chain and a virtual site for each backbone bead.
    assert len(molecule.nodes) == initial_n_backbone * 3


def test_go_virt_includes_no_moltype():
    """
    Test the processor raises an exception if the moltype is not defined.
    """
    molecule = Molecule()
    processor = GoVirtIncludes()
    with pytest.raises(ValueError):
        processor.run_molecule(molecule)
