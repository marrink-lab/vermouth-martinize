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

import pytest
from vermouth import Molecule
from vermouth.processors import GoVirtIncludes

def test_go_virt_includes():
    """
    Test the processor in normal conditions.
    """
    molecule = Molecule()
    molecule.meta['pre_section_lines'] = {
        'virtual_sitesn': ['I was there before'],
    }
    molecule.meta['post_section_lines'] = {
        'atoms': ['I exist'],
    }
    molecule.meta['moltype'] = 'TEST'

    processor = GoVirtIncludes()
    processor.run_molecule(molecule)

    expected_pre = {
        'virtual_sitesn': ['I was there before'],
    }
    expected_post = {
        'atoms': ['I exist', '#include "TEST_atoms_VirtGoSite.itp"'],
        'virtual_sitesn': ['#include "TEST_virtual_sitesn_VirtGoSite.itp"'],
        'exclusions': ['#include "TEST_exclusions_VirtGoSite.itp"'],
    }

    assert molecule.meta['pre_section_lines'] == expected_pre
    assert molecule.meta['post_section_lines'] == expected_post


def test_go_virt_includes_no_moltype():
    """
    Test the processor raises an exception if the moltype is not defined.
    """
    molecule = Molecule()
    processor = GoVirtIncludes()
    with pytest.raises(ValueError):
        processor.run_molecule(molecule)
