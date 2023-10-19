"""
Test for the water bias processor.
"""
import pytest
import numpy as np
import networkx as nx
import vermouth
from vermouth.rcsu.go_vs_includes import VirtualSiteCreator
from vermouth.tests.test_water_bias import create_sys_all_attrs
from vermouth.tests.test_apply_rubber_band import test_molecule
from vermouth.rcsu.go_structure_bias import ComputeStructuralGoBias

def test_compute_go_interaction(test_molecule):
    contacts = [("atom_a", "atom_b", 2.3),
                ("atom_a", "atom_c", 4.1),
                ("atom_q", "atom_p", 0.0)]
    denom = 2**(1/6.)
    expected = {("atom_a", "atom_b"): 2.3/denom,
                ("atom_a", "atom_c"): 4.1/denom,
                ("atom_q", "atom_p"): 0/denom}


    system = vermouth.System()
    system.molecules.append(test_molecule)
    go_processor = ComputeStructuralGoBias(contact_map=None,
                                           cutoff_short=None,
                                           cutoff_long=None,
                                           go_eps=2.1,
                                           res_dist=None,
                                           moltype="molecule_0")
    go_processor.system = system

    go_processor.compute_go_interaction(contacts)
    for nb_params in system.gmx_topology_params['nonbond_params']:
        assert nb_params.atoms in expected
        assert np.isclose(nb_params.sigma, expected[nb_params.atoms])
        assert nb_params.epsilon == 2.1

@pytest.mark.parametrize('cmap, cshort, clong, rdist, expected',(
        # single contact good 
        ([(1, 'A', 4, 'A')],
         0.3,
         2.0,
         0,
         [("mol_0_1", "mol_0_4", 1.5)]),
        # add symmetric contact good 
        ([(1, 'A', 4, 'A'), (4, 'A', 1, 'A')],
         0.3,
         2.0,
         0,
         [("mol_0_1", "mol_0_4", 1.5)]),
        # single contact bad -> cshort 
        ([(1, 'A', 3, 'A')],
         1.5,
         2.0,
         0,
         []),
        # single contact bad -> clong 
        ([(1, 'A', 4, 'A')],
         0.5,
         0.8,
         0,
         []),
        # single contact bad -> rdist 
        ([(1, 'A', 4, 'A')],
         0.3,
         2.0,
         5,
         []),
        # two contacts good 
        ([(1, 'A', 3, 'A'), (1, 'A', 4, 'A')],
         0.3,
         2.0,
         1,
         [("mol_0_1", "mol_0_3", 1.0),
          ("mol_0_1", "mol_0_4", 1.5),
         ]),
        # one good one bad rdist 
        ([(1, 'A', 4, 'A'), (1, 'A', 2, 'A')],
         0.3,
         2.0,
         2,
         [("mol_0_1", "mol_0_4", 1.5)]),
        ))
def test_contact_selector(test_molecule, 
                          cmap, 
                          cshort, 
                          clong, 
                          rdist, 
                          expected):

    # define some possible positions
    pos = np.array([[0.0, 0.0, 0.0],
                    [0.0, 0.4, 0.0],
                    [0.0, 0.8, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.4, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [1.5, 0.4, 0.0],
                    [1.5, 0.8, 0.0]])
    for node in test_molecule.nodes:
        test_molecule.nodes[node]['position'] = pos[node]

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
                                  moltype="mol_0", 
                                  secstruc=secstruc,
                                  defaults={"chain": "A"}, 
                                  attrs={"resname": resnames,
                                         "atype": atypes})

    # generate the virtual sites
    VirtualSiteCreator().run_system(system)
    # initialize the Go processor
    go_processor = ComputeStructuralGoBias(contact_map=cmap,
                                           cutoff_short=cshort,
                                           cutoff_long=clong,
                                           go_eps=2.1,
                                           res_dist=rdist,
                                           moltype="mol_0")
    go_processor.res_graph = vermouth.graph_utils.make_residue_graph(test_molecule)
    # run the contact map selector
    contact_matrix = go_processor.contact_selector(test_molecule)
    assert contact_matrix == expected
