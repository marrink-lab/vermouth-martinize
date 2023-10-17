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

    # set mol meta
    test_molecule.meta['moltype'] = "molecule_0"
    # assign some atomtypes; as the molecule is rather
    # complex we don't need to redo it every time
    atypes = {0: "P1", 1: "SN4a", 2: "SN4a", 
              3: "SP1", 4: "C1",
              5: "TP1", 
              6: "P1", 7: "SN3a", 8: "SP4"}
    nx.set_node_attributes(test_molecule, atypes, "atype")
    sizes = {0: 0.47, 3: 0.41, 5: 0.38, 6: 0.47}
    # assign residue names
    resnames = {0: "A", 1: "A", 2: "A", 
                3: "B", 4: "B",
                5: "C", 
                6: "D", 7: "D", 8: "D"}
    nx.set_node_attributes(test_molecule, resnames, "resname")
    # assign resids
    resids = nx.get_node_attributes(test_molecule, "resid")
    nx.set_node_attributes(test_molecule, resids, "_old_resid")
    # assign chain ids
    nx.set_node_attributes(test_molecule, "A", "chain")
    
    # make the proper force-field
    ff = ForceField("test")
    ff.variables['water_type'] = "W"
    ff.variables['regular'] = 0.47
    ff.variables['small'] = 0.41
    ff.variables['tiny'] = 0.38

    res_graph = vermouth.graph_utils.make_residue_graph(test_molecule)
    for node in res_graph.nodes:
        mol_nodes = res_graph.nodes[node]['graph'].nodes
        block = vermouth.molecule.Block()
        resname = res_graph.nodes[node]['resname']
        resid = res_graph.nodes[node]['resid']
        # assign secondary structure
        for mol_node in mol_nodes:
            test_molecule.nodes[mol_node]['cgsecstruct'] = secstruc[resid]
            block.add_node(test_molecule.nodes[mol_node]['atomname'],
                           atype=test_molecule.nodes[mol_node]['atype'])

        ff.blocks[resname] = block


    # create the system
    test_molecule._force_field = ff
    system = System()
    system.molecules.append(test_molecule)

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
