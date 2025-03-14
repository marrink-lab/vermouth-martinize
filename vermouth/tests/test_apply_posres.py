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
Test the ApplyPosres processor and the related functions.
"""
from vermouth import selectors
from vermouth.processors import apply_posres
from vermouth.tests.helper_functions import test_molecule, create_sys_all_attrs

def test_apply_posres(test_molecule):

    selector =  selectors.select_backbone

    mol = apply_posres.apply_posres(molecule=test_molecule,
                                    selector=selector,
                                    atomnames='BB',
                                    force_constant=1000)

    expected_interaction_sites = [i for i in mol.nodes if selector(mol.nodes[i])]
    posres_atoms = [i.atoms[0] for i in mol.interactions['position_restraints']]

    assert set(expected_interaction_sites) == set(posres_atoms)

def test_ApplyPosres(test_molecule):
    resnames = {0: "ALA", 1: "ALA", 2: "ALA",
                3: "GLY", 4: "GLY",
                5: "MET",
                6: "ARG", 7: "ARG", 8: "ARG"}
    secstruc =  {1: "H", 2: "H", 3: "H", 4: "H"}

    atypes = {0: "P1", 1: "SN4a", 2: "SN4a",
              3: "SP1", 4: "C1",
              5: "TP1",
              6: "P1", 7: "SN3a", 8: "SP4"}

    system = create_sys_all_attrs(test_molecule,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"resname": resnames,
                                         "atype": atypes})

    apply_posres.ApplyPosres((selectors.select_backbone, "BB"), 1000).run_system(system)

    assert all(mol.interactions.get('position_restraints') for mol in system.molecules)
    assert all(mol.meta.get('define').get('POSRES_FC') == 1000 for mol in system.molecules)

