# Copyright 2025 University of Groningen
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
Test some of the behaviour of :class:`vermouth.system.System`
"""

import pytest

from vermouth import System, Molecule
from vermouth.forcefield import ForceField


def test_add_molecule_mismatch_ff():
    """
    Assert that adding a molecule to a system with a mismatching force field
    errors.
    """
    ff_a = ForceField(name='a')
    ff_b = ForceField(name='b')
    assert ff_a != ff_b

    system = System(force_field=ff_a)
    mol_a = Molecule(force_field=ff_a)
    system.add_molecule(mol_a)
    assert len(system.molecules) == 1
    mol_b = Molecule(force_field=ff_b)
    with pytest.raises(KeyError):
        system.add_molecule(mol_b)


def test_system_automatic_ff():
    """
    Assert that a system automatically gets a force field when given a molecule.
    """
    ff_a = ForceField(name='a')
    ff_b = ForceField(name='b')
    assert ff_a != ff_b

    system = System(force_field=None)
    mol_a = Molecule(force_field=ff_a)
    system.add_molecule(mol_a)
    assert len(system.molecules) == 1
    assert system.force_field == ff_a
    mol_b = Molecule(force_field=ff_b)
    with pytest.raises(KeyError):
        system.add_molecule(mol_b)
