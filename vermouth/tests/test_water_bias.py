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
import networkx as nx
import vermouth
from vermouth.rcsu.go_vs_includes import VirtualSideCreator
from vermouth.system import System
from vermouth.forcefield import ForceField
from vermouth.processors.water_bias import ComputeWaterBias
from vermouth.tests.test_apply_rubber_band import test_molecule

def create_sys_all_attrs(molecule, moltype, secstruc, defaults, attrs):
    """
    Generate a test system from a molecule
    with all attributes set and blocks in
    force-field.

    Parameters
    ----------
    molecule: :class:vermouth.Molecule
    moltype: str
        sets meta['moltype']
    secstruc: dict[int, str]
        secondary structure attributes
        as resid str pairs
    defauts: dict
        dict of attribute default value
    attrs: dict
        dict of attribute name and dict
        of node value pairs

    Returns
    -------
    :class:vermouth.System
    """
    # set mol meta
    molecule.meta['moltype'] = moltype
    # assign default node attributes
    for attr, value in defaults.items():
        nx.set_node_attributes(molecule, value, attr)

    # assign node attributes
    for attr, values in attrs.items():
        nx.set_node_attributes(molecule, values, attr)

    # assign resids
    resids = nx.get_node_attributes(molecule, "resid")
    nx.set_node_attributes(molecule, resids, "_old_resid")
    
    # make the proper force-field
    ff = ForceField("test")
    ff.variables['water_type'] = "W"
    ff.variables['regular'] = 0.47
    ff.variables['small'] = 0.41
    ff.variables['tiny'] = 0.38

    res_graph = vermouth.graph_utils.make_residue_graph(molecule)
    for node in res_graph.nodes:
        mol_nodes = res_graph.nodes[node]['graph'].nodes
        block = vermouth.molecule.Block()
        resname = res_graph.nodes[node]['resname']
        resid = res_graph.nodes[node]['resid']
        # assign secondary structure
        for mol_node in mol_nodes:
            molecule.nodes[mol_node]['cgsecstruct'] = secstruc[resid]
            block.add_node(molecule.nodes[mol_node]['atomname'],
                           atype=molecule.nodes[mol_node]['atype'])

        ff.blocks[resname] = block


    # create the system
    molecule._force_field = ff
    system = System()
    system.molecules.append(molecule)
    return system

@pytest.mark.parametrize('secstruc, water_bias, idr_regions, expected',
        ((
        # only auto-bias single sec struct
        {1: "H", 2: "H", 3: "H", 4: "H"},
        {"H": 2.1},
        [],
        {0: "H", 3: "H", 5: "H", 6: "H"}
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
        [(2, 3)],
        {2: "idr", 3: "idr"}
        ),
        # idp and sec struc bias
        ({1: "H", 2: "H", 3: "C", 4: "C"},
        {"idr": 1.1, "C": 3.1, "H": 2.1},
        [(2, 3)],
        {0: "H", 2: "idr", 3: "idr", 4: "C"}
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
    VirtualSideCreator().run_system(system)

    processor = ComputeWaterBias(water_bias=water_bias, 
                                 auto_bias=True, 
                                 idr_regions=idr_regions)
    processor.run_system(system)
    for nb_params in system.gmx_topology_params['nonbond_params']:
        assert nb_params.atoms[0] == "W"
        vs_node = nb_params.atoms[1]
        for vs in system.molecules[0].interactions['virtual_sitesn']:
            if vs.atoms[0] == vs_node:
                bb_node = vs.atoms[1]
                water_eps = water_bias[expected[bb_node]]
                water_sig = sizes[bb_node]
                assert water_eps == nb_params.epsilon
                assert water_sig == nb_params.sigma
