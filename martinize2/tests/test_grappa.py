import collections
import pytest
import networkx as nx

import martinize2.graphing.grappa as grappa


RefEntry = collections.namedtuple(
    'RefEntry', 'string processed_string nodes edges attributes'
)

REFERENCE = {
    'BB': RefEntry(
        'N(H,.) CA(HA,.) C(O1,.) @CA {chiral:(N,C,HA)}',
        'N(H,.) CA(HA,.) C(O1,.) @CA {chiral:(N,C,HA)}',
        ['N', 'HA', 'C', 'H', 'O1', 'CA'],
        [('N', 'CA'), ('N', 'H'), ('HA', 'CA'), ('C', 'CA'), ('C', 'O1')],
        {'CA': {'chiral': '(N,C,HA)', 'stub': 1}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'GLY': RefEntry(
        '<BB> @HA =HA1 @CA HA2',
        '<BB> @HA =HA1 @CA HA2',
        ['N', 'HA1', 'C', 'H', 'HA2', 'O1', 'CA'],
        [('N', 'CA'), ('N', 'H'), ('HA1', 'CA'), ('C', 'CA'),
         ('C', 'O1'), ('HA2', 'CA')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C':{'stub': 1}, 'N': {'stub': 1}},
    ),
    'ALA': RefEntry(
        '<BB> @CA CB(HB[1-3])',
        '<BB> @CA CB(HB[1-3])',
        ['N', 'CB', 'HA', 'C', 'H', 'HB2', 'HB3', 'HB1', 'O1', 'CA'],
        [('N', 'CA'), ('N', 'H'), ('CB', 'HB2'), ('CB', 'HB3'), ('CB', 'CA'),
         ('CB', 'HB1'), ('HA', 'CA'), ('C', 'CA'), ('C', 'O1')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'PHE': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG CD1(HD1) CE1(HE1) CZ(HZ) CE2(HE2) CD2(HD2)',
        '<BB> @CA CB(HB1,HB2) CG CD1(HD1) CE1(HE1) CZ(HZ) CE2(HE2) CD2(HD2)',
        ['HZ', 'CE2', 'C', 'N', 'HD2', 'CD2', 'HD1', 'CD1', 'HB1', 'HE2',
         'O1', 'HB2', 'CB', 'HA', 'CG', 'H', 'CE1', 'HE1', 'CZ', 'CA'],
        [('HZ', 'CZ'), ('CE2', 'HE2'), ('CE2', 'CZ'), ('CE2', 'CD2'),
         ('C', 'CA'), ('C', 'O1'), ('N', 'CA'), ('N', 'H'), ('HD2', 'CD2'),
         ('HD1', 'CD1'), ('CD1', 'CE1'), ('CD1', 'CG'), ('HB1', 'CB'),
         ('HB2', 'CB'), ('CB', 'CA'), ('CB', 'CG'), ('HA', 'CA'),
         ('CE1', 'CZ'), ('CE1', 'HE1')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
}


@pytest.fixture()
def graph_dict():
    names_to_use = ['BB', 'PHE']
    graph_dict = {}
    for name in names_to_use:
        graph_dict[name] = nx.Graph()
        graph_dict[name].add_edges_from(REFERENCE[name].edges)
    return graph_dict


@pytest.mark.parametrize('grappa_string, ref_string', (
    (
        '/#=1-5/C#(O#,H#1,H#2)/ @C5 H53 @O4 -H41 !C1',
        'C1(O1,H11,H12) C2(O2,H21,H22) C3(O3,H31,H32) C4(O4,H41,H42) C5(O5,H51,H52) @C5 H53 @O4 -H41 !C1',
    ),
    pytest.param(
        '/#=1-5/C#(O# H#1, H#2)/ @C5 H53 @O4 -H41 !C1',
        'C1(O1 H11,H12) C2(O2 H21,H22) C3(O3 H31,H32) C4(O4 H41,H42) C5(O5 H51,H52) @C5 H53 @O4 -H41 !C1',
        marks=pytest.mark.xfail(reason='Issue #26'),
    ),
))
def test_preprocess(grappa_string, ref_string):
    assert grappa.preprocess(grappa_string) == ref_string


@pytest.mark.parametrize('name, grappa_string, nodes, edges', (
    (name, ref.string, ref.nodes, ref.edges)
    for name, ref in REFERENCE.items()
))
def test_process_graph(name, grappa_string, nodes, edges, graph_dict):
    residue = grappa.process(grappa_string, graphs=graph_dict)

    # The residues as undirected graphs. When comparing their edges, neither
    # the order of the edges themselves, nor the order of the nodes within an
    # edge should matter. We sort the edges inside and out.
    residue_edges = sorted([sorted(edge) for edge in residue.edges])
    reference_edges = sorted([sorted(edge) for edge in edges])
    assert residue_edges == reference_edges
    assert sorted(residue.nodes) == sorted(nodes)


@pytest.mark.xfail(reason='Issue #27')
@pytest.mark.parametrize('name, grappa_string, attributes', (
    (name, ref.string, ref.attributes)
    for name, ref in REFERENCE.items()
))
def test_process_attributes(name, grappa_string, attributes, graph_dict):
    residue = grappa.process(grappa_string, graphs=graph_dict)

    # Make sure that the expected attributes are created.
    for atom, atom_attributes in attributes.items():
        for key, value in atom_attributes.items():
            assert key in residue.nodes.get(atom, {})
            assert value == residue.nodes[atom][key]

    # Make sure there are no unexpected attributes.
    for node, atom_attributes in residue.nodes.items():
        for key, value in atom_attributes.items():
            assert key in attributes.get(node, {})
            # If the attribute exist for the atom, we checked already that the
            # value was as expected.
