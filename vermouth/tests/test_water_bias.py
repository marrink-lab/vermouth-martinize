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
Test for the water bias processor.
"""
import pytest
import vermouth
from vermouth.rcsu.go_vs_includes import VirtualSiteCreator
from vermouth.processors.water_bias import ComputeWaterBias
from vermouth.tests.helper_functions import create_sys_all_attrs, test_molecule
from vermouth.gmx.topology import NonbondParam

@pytest.mark.parametrize('secstruc, water_bias, idr_regions, expected',
        ((
        # only auto-bias single sec struct
        {1: "H", 2: "H", 3: "H", 4: "H"},
        {"H": 2.1},
        [],
        {0: "H", 3: "H", 5: "H", 6: "H"}
        ),
        # test skip unassigned node
        ({1: None, 2: "H", 3: "H", 4: "H"},
        {"H": 2.1},
        [],
        {3: "H", 5: "H", 6: "H"}
        ),
        # only auto-bias two sec struct
        ({1: "H", 2: "H", 3: "C", 4: "C"},
        {"H": 2.1, "C": 3.1},
        [],
        {0: "H", 3: "H", 5: "C", 6: "C"}
        ),
        # only idp bias
        ({1: "H", 2: "H", 3: "C", 4: "C"},
        {"idr": 2.1},
        ["2:3"],
        {3: "idr", 5: "idr"}
        ),
        # idp and sec struc bias
        ({1: "H", 2: "H", 3: "C", 4: "C"},
        {"idr": 1.1, "C": 3.1, "H": 2.1},
        ["2:3"],
        {0: "H", 3: "idr", 5: "idr", 6: "C"}
        )
        ))
def test_assign_residue_water_bias(test_molecule,
                                   secstruc,
                                   water_bias,
                                   idr_regions,
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

    system = create_sys_all_attrs(test_molecule,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"resname": resnames,
                                         "atype": atypes})

    # generate the virtual sites
    VirtualSiteCreator().run_system(system)

    processor = ComputeWaterBias(water_bias=water_bias,
                                 auto_bias=True,
                                 idr_regions=idr_regions)
    processor.run_system(system)
    for nb_params in system.gmx_topology_params['nonbond_params']:
        assert nb_params.atoms[0] == "W"
        vs_node_atype = nb_params.atoms[1]
        for atom in system.molecules[0].atoms:
            if atom[1]['atype'] == vs_node_atype:
                vs_node = atom[0]
        for vs in system.molecules[0].interactions['virtual_sitesn']:
            if vs.atoms[0] == vs_node:
                bb_node = vs.atoms[1]
                water_eps = water_bias[expected[bb_node]]
                water_sig = sizes[bb_node]
                assert water_eps == nb_params.epsilon
                assert water_sig == nb_params.sigma

@pytest.mark.parametrize('idr_regions, go_bonds, expected',
         (
            (#no idrs to remove
             [],
             [[1, 3], [2, 4]],
             [[1, 3], [2, 4]]
            ),
            (#central idr to remove bonds from
             ["2:3"],
             [[1, 3], [2, 4], [1, 4]],
             [[1, 4]]
            ),
            (#remove all bonds within a region
            ["1:4"],
            [[1, 2], [2, 3], [3, 4], [1, 4]],
            []
            )
          ))
def test_cross_go_bond_removal(test_molecule,
                               idr_regions,
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

    #apply water bias and remove folded-disordered domain bonds
    processor = ComputeWaterBias(water_bias={"idr": 1.1, "C": 3.1, "H": 2.1}, # doesn't matter
                                 auto_bias=True,
                                 idr_regions=idr_regions)
    processor.run_system(system)

    #find the go bonds which remain after removal, and don't involve water
    remaining = [list(i.atoms) for i in system.gmx_topology_params["nonbond_params"] if 'W' not in list(i.atoms)]

    assert len(remaining) == len(expected)
    for i in expected:
        expected_names = [f"molecule_0_{j}" for j in i]

        assert expected_names in remaining

def test_no_moltype_error(test_molecule):
    """
    Test that various high level errors are
    properly raised.
    """
    # set up processor
    processor = ComputeWaterBias(water_bias={"C": 3.1},
                                 auto_bias=True,
                                 idr_regions=[])
    # no moltype set
    system = vermouth.System()
    system.add_molecule(test_molecule)
    with pytest.raises(ValueError):
        processor.run_system(system)

def test_no_system_error(test_molecule):
    """
    Test that various high level errors are
    properly raised.
    """
    # set up processor
    processor = ComputeWaterBias(water_bias={"C": 3.1},
                                 auto_bias=True,
                                 idr_regions=[])
    test_molecule.meta['moltype'] = "random"
    # no system
    with pytest.raises(IOError):
        processor.run_molecule(test_molecule)

def test_clean_return(test_molecule):
    # set up processor
    processor = ComputeWaterBias(water_bias={"C": 3.1},
                                 auto_bias=None,
                                 idr_regions=[])
    test_molecule.meta['moltype'] = "random"
    system = vermouth.System()
    system.add_molecule(test_molecule)
    assert processor.run_system(system) == system
