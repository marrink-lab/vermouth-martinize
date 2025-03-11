"""
Test for the water bias processor.
"""
import pytest
import numpy as np
import networkx as nx
import vermouth
from vermouth.rcsu.go_vs_includes import VirtualSiteCreator
from vermouth.tests.test_water_bias import create_sys_all_attrs
from vermouth.tests.helper_functions import test_molecule
from vermouth.rcsu.go_structure_bias import ComputeStructuralGoBias

def test_compute_go_interaction(test_molecule):
    contacts = [("atom_a", "atom_b", 2.3),
                ("atom_a", "atom_c", 4.1),
                ("atom_q", "atom_p", 0.0)]
    denom = 2**(1/6.)
    expected = {(k1, k2): v/denom for k1, k2, v in contacts}


    system = vermouth.System()
    system.add_molecule(test_molecule)
    go_processor = ComputeStructuralGoBias(cutoff_short=None,
                                           cutoff_long=None,
                                           go_eps=2.1,
                                           res_dist=None,
                                           moltype="molecule_0")
    go_processor.system = system

    go_processor.compute_go_interaction(contacts)
    for nb_params in system.gmx_topology_params['nonbond_params']:
        assert nb_params.atoms in expected
        assert pytest.approx(expected[nb_params.atoms]) == nb_params.sigma
        assert nb_params.epsilon == 2.1

@pytest.mark.parametrize('cmap, cshort, clong, rdist, expected',(
        # single assymetric contact bad
        ([(1, 'A', 4, 'A')],
         0.3,
         2.0,
         0,
         []),
        # single symmetric contact good
        ([(1, 'A', 4, 'A'), (4, 'A', 1, 'A')],
         0.3,
         2.0,
         0,
         [("mol_0_4", "mol_0_1", 1.5)]),
        # single symmetric contact bad -> cshort
        ([(3, 'A', 1, 'A'),(1, 'A', 3, 'A')],
         1.5,
         2.0,
         0,
         []),
        # single symmetric contact bad -> clong
        ([(4, 'A', 1, 'A'), (1, 'A', 4, 'A')],
         0.5,
         0.8,
         0,
         []),
        # single symmetric contact bad -> rdist
        ([(4, 'A', 1, 'A'), (1, 'A', 4, 'A')],
         0.3,
         2.0,
         5,
         []),
        # single contact bad -> cshort & assym
        ([(1, 'A', 3, 'A')],
         1.5,
         2.0,
         0,
         []),
        # single contact bad -> clong & assym
        ([(1, 'A', 4, 'A')],
         0.5,
         0.8,
         0,
         []),
        # single contact bad -> rdist & assym
        ([(1, 'A', 4, 'A')],
         0.3,
         2.0,
         5,
         []),
        # two symmetric contacts good
        ([(3, 'A', 1, 'A'), (4, 'A', 1, 'A'),(1, 'A', 3, 'A'), (1, 'A', 4, 'A')],
         0.3,
         2.0,
         1,
         [("mol_0_1", "mol_0_3", 1.0),
          ("mol_0_1", "mol_0_4", 1.5),
         ]),
        # one symmetric contacts good, one assymetric contact bad
        ([(3, 'A', 1, 'A'), (4, 'A', 1, 'A'),(1, 'A', 3, 'A')],
         0.3,
         2.0,
         1,
         [("mol_0_1", "mol_0_3", 1.0)]),
        # one good one bad rdist
        ([(4, 'A', 1, 'A'), (2, 'A', 1, 'A'), (1, 'A', 4, 'A'), (1, 'A', 2, 'A')],
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
    test_molecule.force_field.macros['bb_atomname'] = 'BB'

    # generate the virtual sites
    VirtualSiteCreator().run_system(system)
    # add the contacts to the system
    system.go_params["go_map"] = [cmap]
    # initialize the Go processor
    go_processor = ComputeStructuralGoBias(cutoff_short=cshort,
                                           cutoff_long=clong,
                                           go_eps=2.1,
                                           res_dist=rdist,
                                           moltype="mol_0",
                                           system=system)
    go_processor.res_graph = vermouth.graph_utils.make_residue_graph(test_molecule)
    # run the contact map selector
    contact_matrix = go_processor.contact_selector(test_molecule)
    assert contact_matrix == expected



@pytest.mark.parametrize('cmap, expected',(
        # single symmetric contact good chain id
        ([(1, 'A', 4, 'A'), (4, 'A', 1, 'A')],
         False),
        # single symmetric contact bad chain id
        ([(1, 'Z', 4, 'Z'), (4, 'Z', 1, 'Z')],
         True),
))
def test_correct_chains(test_molecule, cmap, expected, caplog):

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
    test_molecule.force_field.macros['bb_atomname'] = 'BB'

    # generate the virtual sites
    VirtualSiteCreator().run_system(system)
    # add the contacts to the system
    system.go_params["go_map"] = [cmap]
    # initialize the Go processor
    go_processor = ComputeStructuralGoBias(cutoff_short=0.3,
                                           cutoff_long=2.0,
                                           go_eps=2.1,
                                           res_dist=0,
                                           moltype="mol_0",
                                           system=system)

    caplog.clear()
    go_processor.run_system(system)

    if expected:
        assert any(rec.levelname == 'WARNING' for rec in caplog.records)
        # makes sure the warning is only printed once
        assert len(caplog.records) == 1
    else:
        assert caplog.records == []


