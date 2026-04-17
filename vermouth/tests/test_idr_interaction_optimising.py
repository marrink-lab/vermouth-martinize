# Copyright 2026 University of Groningen
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
import pytest
import vermouth
from vermouth.rcsu.go_vs_includes import VirtualSiteCreator
from vermouth.processors.water_bias import ComputeWaterBias
from vermouth.processors.idr_interaction_optimising import IDRInteractionOptimising
from vermouth.tests.helper_functions import create_sys_all_attrs, test_molecule
from vermouth.gmx.topology import NonbondParam
from vermouth.molecule import Interaction
import networkx

@pytest.mark.parametrize('id_regions, go_bonds, expected',
         (
            (#no idrs to remove
             [],
             [[1, 3], [2, 4]],
             [[1, 3], [2, 4]]
            ),
            (#central idr to remove bonds from
             ["A-2:3"],
             [[1, 3], [2, 4], [1, 4]],
             [[1, 4]]
            ),
            (#remove all bonds within a region
            ["A-1:4"],
            [[1, 2], [2, 3], [3, 4], [1, 4]],
            []
            )
          ))
def test_cross_go_bond_removal(test_molecule,
                               id_regions,
                               go_bonds,
                               expected):
    # bead sizes
    sizes = {0: 0.47, 3: 0.41, 5: 0.38, 6: 0.47}
    # the molecule atomtypes
    atypes = {0: "P1", 1: "SN4a", 2: "SN4a",
              3: "SP1", 4: "C1",
              5: "TP1",
              6: "P1", 7: "SN3a", 8: "SP4"}
    # the molecule resnames
    resnames = {0: "A", 1: "A", 2: "A",
                3: "B", 4: "B",
                5: "C",
                6: "D", 7: "D", 8: "D"}
    secstruc = {1: "H", 2: "H", 3: "H", 4: "H"}

    system = create_sys_all_attrs(test_molecule,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"resname": resnames,
                                         "atype": atypes})

    # generate the virtual sites
    VirtualSiteCreator().run_system(system)

    #add the go bonds between residues
    for go_bond in go_bonds:
        atypes = []
        for index in go_bond:
            for atom in [i for i in system.molecules[0].atoms]:
                if (atom[1]['resid'] == index) and (atom[1]['atomname'] == 'CA'):
                    atypes.append(atom[1]['atype'])
        contact_bias = NonbondParam(atoms=(atypes[0], atypes[1]),
                                sigma=0.5, #these values don't matter
                                epsilon=0.5,
                                meta={"comment": ["go bond"]})
        system.gmx_topology_params["nonbond_params"].append(contact_bias)

    # remove folded-disordered domain Go interactions
    processor = IDRInteractionOptimising(go=True,
                                         elastic=False,
                                         id_regions=id_regions)
    processor.run_system(system)

    #find the go bonds which remain after removal, and don't involve water
    remaining = [list(i.atoms) for i in system.gmx_topology_params["nonbond_params"] if 'W' not in list(i.atoms)]

    assert len(remaining) == len(expected)
    for i in expected:
        expected_names = [f"molecule_0_{j}" for j in i]

        assert expected_names in remaining
 
 
@pytest.mark.parametrize('id_regions, el_bonds, expected',
         (
            (#no idrs to remove
             [],
             [[1, 6], [4, 7]],
             [[1, 6], [4, 7]]
            ),
            (#central idr to remove bonds from
             ["A-2:3"],
             [[1, 6], [4, 7], [1, 7]],
             [[1, 7]]
            ),
            (#remove all bonds within a region
            ["A-1:4"],
            [[1, 4], [4, 6], [6, 7], [1, 7]],
            []
            )
          ))
def test_cross_el_bond_removal(test_molecule,
                               id_regions,
                               el_bonds,
                               expected):
    # bead sizes
    sizes = {0: 0.47, 3: 0.41, 5: 0.38, 6: 0.47}
    # the molecule atomtypes
    atypes = {1: "P1", 2: "SN4a", 3: "SN4a",
              4: "SP1", 5: "C1",
              6: "TP1",
              7: "P1", 8: "SN3a", 9: "SP4"}
    # the molecule resnames
    resnames = {1: "A", 2: "A", 3: "A",
                4: "B", 5: "B",
                6: "C",
                7: "D", 8: "D", 9: "D"}
    stash = {'resid':{1: 1, 2: 1, 3: 1,
                      4: 2, 5: 2,
                      6: 3,
                      7: 4, 8: 4, 9: 4}}
    secstruc = {1: "H", 2: "H", 3: "H", 4: "H"}

    # atom indices actually start from 1 and we use them when processing elastic networks
    G = networkx.relabel_nodes(test_molecule, lambda x: x + 1)
    system = create_sys_all_attrs(G,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"resname": resnames,
                                         "atype": atypes,
                                         "stash": stash})
    #add the elastic bonds between BB beads
    for el_bond in el_bonds:
        system.molecules[0].add_interaction(type_ = 'bonds',
                                            atoms = tuple(el_bond),
                                            parameters = ['1', '10', '5000']
                                            )

    # remove folded-disordered domain bonds
    processor = IDRInteractionOptimising(go=False,
                                         elastic=True,
                                         id_regions=id_regions,
                                         elastic_res_distance=1)
    processor.run_system(system)
    print(system.molecules[0].interactions['bonds'])
    # assert False
    remaining = [i.atoms for i in system.molecules[0].interactions['bonds']]

    assert len(remaining) == len(expected)
    for i in expected:
        assert tuple(i) in remaining
        
 
 