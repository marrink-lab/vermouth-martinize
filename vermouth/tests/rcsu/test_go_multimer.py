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
"""
Test for the multimer Go model processor.
"""
import numpy as np
import pytest
from vermouth import System
from vermouth.molecule import Molecule
from vermouth.graph_utils import make_residue_graph
from vermouth.processors import IDRCrossDomainInteractionRemoval
from vermouth.rcsu.go_utils import get_go_type_from_attributes
from vermouth.rcsu.go_vs_includes import VirtualSiteCreator
from vermouth.rcsu.go_multimer import ComputeMultimerGoBias, check_complete_chains, read_nonbond_params

NONBOND_PARAMS = {frozenset(['P2']): (0.47, 4.06),
                  frozenset(['P2', 'Q5']): (0.47, 5.0)}


def make_molecule(resnames, chain, moltype, x, bb_type='P2', charge=0, z=0.0):
    """
    A chain of backbone beads along y at position (`x`, `z`), with resids
    from 1.
    """
    molecule = Molecule()
    molecule.meta['moltype'] = moltype
    for resid, resname in enumerate(resnames, start=1):
        molecule.add_node(resid, resid=resid, resname=resname, atomname='BB', chain=chain,
                          atype=bb_type, charge=charge, charge_group=resid,
                          position=np.array([x, 0.35 * (resid - 1), z]),
                          stash={'resid': resid})
        if resid > 1:
            molecule.add_edge(resid - 1, resid)
    return molecule


def run_go(molecules, go_map, **kwargs):
    """
    Add the Go virtual sites and the multimer Go model to `molecules`.
    """
    system = System()
    for molecule in molecules:
        system.add_molecule(molecule)
    system.go_params['go_map'] = [go_map]
    VirtualSiteCreator().run_system(system)
    ComputeMultimerGoBias(cutoff_short=0.3, cutoff_long=1.1, go_eps=9.0,
                          go_eps_inter=8.0, go_eps_hetero=7.0, res_dist=2,
                          nonbond_params=NONBOND_PARAMS, **kwargs).run_system(system)
    return system


def symmetric(*contacts):
    """
    A contact map with every contact in both directions.
    """
    return [contact for resid_a, chain_a, resid_b, chain_b in contacts
            for contact in ((resid_a, chain_a, resid_b, chain_b),
                            (resid_b, chain_b, resid_a, chain_a))]


def test_go_multimer(caplog):
    """
    Two copies of one molecule and one other molecule, with an intra contact,
    a homo-inter contact, a self-pair, a hetero contact, and contacts that are
    not symmetric or too close along the residue graph.
    """
    caplog.set_level('INFO')
    copy_a = make_molecule('AGVL', 'A', 'mol_0', 0.0)
    copy_b = make_molecule('AGVL', 'B', 'mol_0', 0.5)
    other = make_molecule('KK', 'C', 'mol_1', 1.0, bb_type='Q5')
    other.nodes[1]['position'] = np.array([1.0, 1.05, 0.0])
    go_map = symmetric((1, 'A', 4, 'A'), (1, 'B', 4, 'B'), (2, 'A', 3, 'B'),
                       (2, 'A', 2, 'B'), (4, 'B', 1, 'C'), (1, 'A', 3, 'A'))
    go_map.append((1, 'A', 1, 'B'))
    system = run_go([copy_a, copy_b, other], go_map)

    factor = 2**(1/6)
    expected = {('mol_0_1', 'mol_0_4'): (1.05 / factor, 9.0),
                ('mol_0_1_B', 'mol_0_4_B'): (1.05 / factor, -9.0),
                ('mol_0_2_C', 'mol_0_3_C'): (np.hypot(0.5, 0.35) / factor, 8.0),
                ('mol_0_2_D', 'mol_0_3_D'): (0.47, -4.06),
                ('mol_0_2_C', 'mol_0_2_C'): (0.5 / factor, 8.0),
                ('mol_0_2_D', 'mol_0_2_D'): (0.47, -4.06),
                ('mol_0_4_C', 'mol_1_1_C'): (0.5 / factor, 7.0),
                ('mol_0_4_D', 'mol_1_1_D'): (0.47, -5.0)}
    found = {param.atoms: (param.sigma, param.epsilon)
             for param in system.gmx_topology_params['nonbond_params']}
    assert len(system.gmx_topology_params['nonbond_params']) == len(expected)
    assert found.keys() == expected.keys()
    for atoms, params in expected.items():
        assert found[atoms] == pytest.approx(params)

    atomtypes = [atomtype.molecule.nodes[atomtype.node]['atype']
                 for atomtype in system.gmx_topology_params['atomtypes']]
    new_atomtypes = [atype for atype in atomtypes if atype.count('_') == 3]
    assert sorted(new_atomtypes) == sorted(['mol_0_1_B', 'mol_0_4_B', 'mol_0_2_C', 'mol_0_3_C',
                                            'mol_0_4_C', 'mol_0_2_D', 'mol_0_3_D', 'mol_0_4_D',
                                            'mol_1_1_C', 'mol_1_1_D'])

    for molecule in (copy_a, copy_b):
        excluded = {(molecule.nodes[atom_a]['atype'], molecule.nodes[atom_b]['atype'],
                     exclusion.meta['group'])
                    for exclusion in molecule.interactions['exclusions']
                    for atom_a, atom_b in [exclusion.atoms]}
        assert excluded == {('P2', 'P2', 'Go model exclusion'),
                            ('mol_0_1_B', 'mol_0_4_B', 'Go model exclusion VWB'),
                            ('mol_0_2_C', 'mol_0_3_C', 'Go model exclusion VWC'),
                            ('mol_0_2_D', 'mol_0_3_D', 'Go model exclusion VWD')}
    assert not other.interactions['exclusions']
    # the copies still share one topology
    assert copy_a.share_moltype_with(copy_b)
    assert 'M3_GO_OLIGOMERS' in copy_a.citations
    assert ("mol_0: 2 copies, 1 intra-, 2 homo-inter- and 1 hetero-inter-molecular Go "
            "contacts; added 2 B, 3 C and 3 D sites per copy.") in caplog.text
    assert ("mol_1: 1 copy, 0 intra-, 0 homo-inter- and 1 hetero-inter-molecular Go "
            "contacts; added 0 B, 1 C and 1 D sites per copy.") in caplog.text


def test_go_multimer_intra_and_inter():
    """
    A residue pair that is both an intra and an inter contact gets the lines
    and exclusions of both.
    """
    copy_a = make_molecule('AGVL', 'A', 'mol_0', 0.0)
    copy_b = make_molecule('AGVL', 'B', 'mol_0', 0.2)
    system = run_go([copy_a, copy_b], symmetric((1, 'A', 4, 'A'), (1, 'A', 4, 'B')))

    assert {param.atoms for param in system.gmx_topology_params['nonbond_params']} == {
        ('mol_0_1', 'mol_0_4'), ('mol_0_1_B', 'mol_0_4_B'),
        ('mol_0_1_C', 'mol_0_4_C'), ('mol_0_1_D', 'mol_0_4_D')}
    groups = [exclusion.meta['group'] for exclusion in copy_a.interactions['exclusions']]
    assert groups == ['Go model exclusion', 'Go model exclusion VWB',
                      'Go model exclusion VWC', 'Go model exclusion VWD']


def test_go_multimer_same_sequence(caplog):
    """
    Molecules with the same sequence but their own itps (e.g. with -sep):
    their inter contacts get the homo epsilon and, as the molecules do not
    share atomtypes, no C/D exclusions.
    """
    caplog.set_level('INFO')
    mol_a = make_molecule('AGVL', 'A', 'mol_0', 0.0)
    mol_b = make_molecule('AGVL', 'B', 'mol_1', 0.5)
    system = run_go([mol_a, mol_b], symmetric((2, 'A', 3, 'B'), (2, 'A', 2, 'B')))

    factor = 2**(1/6)
    expected = {('mol_0_2_C', 'mol_1_3_C'): (np.hypot(0.5, 0.35) / factor, 8.0),
                ('mol_0_2_D', 'mol_1_3_D'): (0.47, -4.06),
                ('mol_0_2_C', 'mol_1_2_C'): (0.5 / factor, 8.0),
                ('mol_0_2_D', 'mol_1_2_D'): (0.47, -4.06)}
    found = {param.atoms: (param.sigma, param.epsilon)
             for param in system.gmx_topology_params['nonbond_params']}
    assert found.keys() == expected.keys()
    for atoms, params in expected.items():
        assert found[atoms] == pytest.approx(params)
    assert not mol_a.interactions['exclusions']
    assert not mol_b.interactions['exclusions']
    assert ("mol_1: 1 copy, 0 intra-, 2 homo-inter- and 0 hetero-inter-molecular Go "
            "contacts; added 0 B, 2 C and 2 D sites per copy.") in caplog.text


def test_go_multimer_missing_nonbond_params():
    """
    Site D needs the backbone parameters from the force field.
    """
    molecules = [make_molecule('AG', 'A', 'mol_0', 0.0, bb_type='P5'),
                 make_molecule('AG', 'B', 'mol_0', 0.5, bb_type='P5')]
    with pytest.raises(KeyError):
        run_go(molecules, symmetric((1, 'A', 1, 'B')))


def test_go_multimer_idr():
    """
    Site A is still found by its attributes, and an IDR residue loses its
    Go bonds (sites A, B and C) but keeps site D.
    """
    copy_a = make_molecule('AGVL', 'A', 'mol_0', 0.0)
    copy_b = make_molecule('AGVL', 'B', 'mol_0', 0.5)
    go_map = symmetric((1, 'A', 4, 'A'), (1, 'A', 3, 'B'), (2, 'A', 3, 'B'))
    system = run_go([copy_a, copy_b], go_map)
    graph = make_residue_graph(copy_a).nodes[0]['graph']
    assert next(get_go_type_from_attributes(graph, resid=1, chain='A', go_site=None,
                                            prefix='mol_0')) == 'mol_0_1'
    assert len(system.gmx_topology_params['nonbond_params']) == 6

    IDRCrossDomainInteractionRemoval(go=True, elastic=False, id_regions=['A-1:1']).run_system(system)

    assert {param.atoms for param in system.gmx_topology_params['nonbond_params']} == {
        ('mol_0_1_D', 'mol_0_3_D'), ('mol_0_2_C', 'mol_0_3_C'), ('mol_0_2_D', 'mol_0_3_D')}


def test_go_multimer_warnings(caplog):
    """
    Warn about contacts whose distance differs between copies, and about
    inter contacts between two charged backbone beads, which get no Go bond.
    """
    copy_a = make_molecule('AGVL', 'A', 'mol_0', 0.0, charge=1)
    copy_b = make_molecule('AGVL', 'B', 'mol_0', 0.5, charge=1)
    copy_b.nodes[4]['position'] = np.array([0.5, 0.85, 0.0])
    go_map = symmetric((1, 'A', 4, 'A'), (1, 'B', 4, 'B'), (2, 'A', 2, 'B'))
    system = run_go([copy_a, copy_b], go_map)

    assert "mol_0_1 - mol_0_4 differs by 0.200 nm" in caplog.text
    assert "mol_0_2 - mol_0_2 joins two charged backbone beads" in caplog.text
    # both have their own type, which martinize2 counts once
    assert {record.type for record in caplog.records if record.levelname == 'WARNING'} == {
        'go-contact-spread', 'go-charged-contact'}
    assert {param.atoms for param in system.gmx_topology_params['nonbond_params']} == {
        ('mol_0_1', 'mol_0_4'), ('mol_0_1_B', 'mol_0_4_B')}
    assert not any(atom.get('go_site') in ('C', 'D') for atom in copy_a.nodes.values())


@pytest.mark.parametrize('intersection', (False, True))
def test_go_multimer_pooling(intersection):
    """
    An intra contact found in one copy is only used with union. An inter
    contact takes its sigma from the pairs of molecules in which it is
    closest, the neighbours A-B and B-C, and not from A-C. With
    intersection, it must be found in both neighbours.
    """
    copies = [make_molecule('AGVL', chain, 'mol_0', x) for chain, x in zip('ABC', (0.0, 0.5, 1.0))]
    # 1-4 in A only; 2-3 between the neighbours A-B and B-C and between A-C; 1-2 only between A-B
    go_map = symmetric((1, 'A', 4, 'A'), (2, 'A', 3, 'B'), (2, 'B', 3, 'C'), (2, 'A', 3, 'C'),
                       (1, 'A', 2, 'B'))
    system = run_go(copies, go_map, intersection=intersection)

    found = {param.atoms: param.sigma for param in system.gmx_topology_params['nonbond_params']
             if not param.atoms[0].endswith(('_B', '_D'))}
    expected = {('mol_0_2_C', 'mol_0_3_C'): np.hypot(0.5, 0.35) / 2**(1/6)}
    if not intersection:
        expected[('mol_0_1', 'mol_0_4')] = 1.05 / 2**(1/6)
        expected[('mol_0_1_C', 'mol_0_2_C')] = np.hypot(0.5, 0.35) / 2**(1/6)
    assert found == pytest.approx(expected)


@pytest.mark.parametrize('intersection', (False, True))
def test_go_multimer_pooling_two_itps(intersection):
    """
    A contact is only pooled over the pairs of molecules of its own two itps.
    X has the copies A and B, Y the copies C, D and E. A-B (X-X), C-D (Y-Y)
    and A-E (X-Y) are all 0.5 nm apart, A-C and B-D (X-Y) 1.0 nm.
    """
    molecules = [make_molecule('AGVL', 'A', 'mol_0', 0.0), make_molecule('AGVL', 'B', 'mol_0', 0.5),
                 make_molecule('GAVL', 'C', 'mol_1', 0.0, z=1.0), make_molecule('GAVL', 'D', 'mol_1', 0.5, z=1.0),
                 make_molecule('GAVL', 'E', 'mol_1', -0.5)]
    # X-X 2-3, Y-Y 1-2 and X-Y 2-2 each in their only pair 0.5 nm apart; X-Y 1-1 in A-C and B-D;
    # X-Y 3-3 in A-C only; intra 1-4 in A only and in all copies of Y
    go_map = symmetric((2, 'A', 3, 'B'), (1, 'C', 2, 'D'), (2, 'A', 2, 'E'), (1, 'A', 1, 'C'),
                       (1, 'B', 1, 'D'), (3, 'A', 3, 'C'), (1, 'A', 4, 'A'), (1, 'C', 4, 'C'),
                       (1, 'D', 4, 'D'), (1, 'E', 4, 'E'))
    system = run_go(molecules, go_map, intersection=intersection)

    found = {param.atoms for param in system.gmx_topology_params['nonbond_params']
             if not param.atoms[0].endswith(('_B', '_D'))}
    expected = {('mol_0_2_C', 'mol_0_3_C'), ('mol_1_1_C', 'mol_1_2_C'), ('mol_0_2_C', 'mol_1_2_C'),
                ('mol_0_1_C', 'mol_1_1_C'), ('mol_1_1', 'mol_1_4')}
    if not intersection:
        expected |= {('mol_0_3_C', 'mol_1_3_C'), ('mol_0_1', 'mol_0_4')}
    assert found == expected


@pytest.mark.parametrize('intersection', (False, True))
def test_select_inter_contacts(caplog, intersection):
    """
    An inter contact is pooled over the pairs of molecules at about the
    distance of those in which it is (nearly) closest. Two stacks side by
    side: A-B and C-D are 0.5 nm apart, A-C and B-D 1.0 nm and A-D 1.12 nm.
    """
    copies = [make_molecule('AGVL', chain, 'mol_0', x, z=z)
              for chain, x, z in zip('ABCD', (0.0, 0.5, 0.0, 0.5), (0.0, 0.0, 1.0, 1.0))]
    system = System()
    for molecule in copies:
        system.add_molecule(molecule)
    processor = ComputeMultimerGoBias(cutoff_short=0.3, cutoff_long=1.1, go_eps=9.0,
                                      go_eps_inter=8.0, go_eps_hetero=7.0, res_dist=2,
                                      nonbond_params=NONBOND_PARAMS, intersection=intersection)
    processor.system = system
    processor.backbone_atoms = {('mol_0', resid): atom for resid, atom in copies[0].nodes.items()}
    pairs = {'AB': (0, 1), 'CD': (2, 3), 'AC': (0, 2), 'BD': (1, 3), 'AD': (0, 3)}
    contacts = {
        # within the stacks
        (2, 3): {'AB': 0.60, 'CD': 0.62},
        # between the stacks
        (1, 1): {'AC': 0.80, 'BD': 0.82},
        # in one pair within the stacks only
        (4, 4): {'AB': 0.70},
        # closest between the stacks, although those molecules are further apart
        (1, 2): {'AB': 1.00, 'CD': 1.02, 'AC': 0.60, 'BD': 0.62},
        # as close within as between the stacks, but not found between B-D
        (1, 3): {'AB': 0.70, 'CD': 0.72, 'AC': 0.72},
        # as close within as between the stacks; A-D is not used
        (2, 4): {'AB': 0.50, 'CD': 0.52, 'AC': 0.53, 'BD': 0.53, 'AD': 1.05},
    }
    selected = processor.select_inter_contacts(
        {(('mol_0', resid_a), ('mol_0', resid_b)): [(pairs[name], dist) for name, dist in found.items()]
         for (resid_a, resid_b), found in contacts.items()})

    expected = {(2, 3): [0.60, 0.62], (1, 1): [0.80, 0.82], (1, 2): [0.60, 0.62],
                (2, 4): [0.50, 0.52, 0.53, 0.53]}
    if not intersection:
        expected.update({(4, 4): [0.70], (1, 3): [0.70, 0.72, 0.72]})
    assert {(end_a[1], end_b[1]): sorted(dists) for (end_a, end_b), dists in selected.items()} == expected
    assert {record.type for record in caplog.records} == {'go-contact-shells'}
    assert ("mol_0_2 - mol_0_4 is about equally short between pairs of molecules "
            "0.500 to 1.000 nm apart") in caplog.text
    assert ("mol_0_1 - mol_0_3 is about" in caplog.text) is not intersection


def test_go_multimer_repeated_chain():
    """
    Residues are identified by chain and resid, so chain IDs must be unique.
    """
    molecules = [make_molecule('AG', 'A', 'mol_0', 0.0), make_molecule('AG', 'A', 'mol_0', 0.5)]
    with pytest.raises(ValueError):
        run_go(molecules, [])


@pytest.mark.parametrize('chains, resnames_b, error', (
    # complete chains with unique chain IDs
    ('AB', ('ALA', 'GLY'), False),
    # a chain split over two molecules, or a repeated chain ID
    ('AA', ('ALA', 'GLY'), True),
    # molecules that are not proteins are not checked
    ('AA', ('W', 'W'), False),
))
def test_check_complete_chains(chains, resnames_b, error):
    """
    Every protein chain must be a single molecule.
    """
    system = System()
    system.add_molecule(make_molecule(('ALA', 'GLY'), chains[0], 'mol_0', 0.0))
    system.add_molecule(make_molecule(resnames_b, chains[1], 'mol_1', 0.5))
    if error:
        with pytest.raises(ValueError):
            check_complete_chains(system)
    else:
        check_complete_chains(system)


def test_read_nonbond_params(tmp_path):
    """
    Only the nonbond_params are read; comments and directives are skipped.
    """
    path = tmp_path / 'ff.itp'
    path.write_text("[ atomtypes ]\n"
                    "P2 72.0 0.000 A 0.0 0.0\n"
                    "[ nonbond_params ]\n"
                    "; comment\n"
                    "  P2  P2  1 4.700000e-01    4.060000e+00 ; self\n"
                    "Q5 P2 1 0.47 5.0\n"
                    "#ifdef GO_VIRT\n"
                    '#include "go_nbparams.itp"\n'
                    "#endif\n")
    assert read_nonbond_params(path) == NONBOND_PARAMS
