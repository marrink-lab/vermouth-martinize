# Copyright 2022 University of Groningen
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
Test for the virtual side creator.
"""
import pytest
import vermouth
from vermouth.rcsu.go_vs_includes import VirtualSiteCreator
from vermouth.tests.helper_functions import test_molecule

def test_no_moltype_error(test_molecule):
    """
    Test that various high level IOErrors are
    properly raised.
    """
    # set up processor
    processor = VirtualSiteCreator()
    # no moltype set
    system = vermouth.System()
    system.molecules.append(test_molecule)
    with pytest.raises(ValueError):
        processor.run_system(system)

def test_no_system_error(test_molecule):
    """
    Test that various high level errors are
    properly raised.
    """
    # set up processor
    processor = VirtualSiteCreator()
    test_molecule.meta['moltype'] = "random"
    # no system
    with pytest.raises(IOError):
        processor.run_molecule(test_molecule)

def test_return_no_nodes():
    mol = vermouth.Molecule()
    processor = VirtualSiteCreator()
    mol.meta['moltype'] = "random"
    system = vermouth.System()
    system.add_molecule(mol)
    assert processor.run_system(system) is None
