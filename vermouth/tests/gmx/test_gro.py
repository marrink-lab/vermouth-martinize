# -*- coding: utf-8 -*-
# Copyright 2018 University of Groningen
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
Unit tests for the GRO reader.
"""
# Pylint complains about hypothesis strategies not receiving a value for the
# `draw` parameter. This is because the `draw` parameter is implicitly filled
# by hypothesis. The pylint warning is disabled for the file instead of being
# disabled at every call of a strategy.
# pylint: disable=no-value-for-parameter

from pprint import pprint
import copy
import collections
import itertools

import numpy as np
import networkx as nx

import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

import vermouth
from vermouth.file_writer import DeferredFileWriter
from vermouth.utils import are_different
from vermouth.molecule import Molecule
from vermouth.gmx import gro

# pylint: disable=redefined-outer-name


# The data comes from residues 3 to 5 of 1BTA.pdb. The atoms from ILE 5 are
# changed in solvent.
GRO_CONTENT = """\
    1ALA      N    1
    1ALA     CA    2
    1ALA      C    3
    1ALA      O    4
    1ALA     CB    5
    1ALA      H    6
    1ALA     HA    7
    1ALA    HB1    8
    1ALA    HB2    9
    1ALA    HB3   10
    2VAL      N   11
    2VAL     CA   12
    2VAL      C   13
    2VAL      O   14
    2VAL     CB   15
    2VAL    CG1   16
    2VAL    CG2   17
    2VAL      H   18
    2VAL     HA   19
    2VAL     HB   20
    2VAL   HG11   21
    2VAL   HG12   22
    2VAL   HG13   23
    2VAL   HG21   24
    2VAL   HG22   25
    2VAL   HG23   26
    3SOL     OW   27
    3SOL    HW1   28
    3SOL    HW2   29
    4SOL     OW   30
    4SOL    HW1   31
    4SOL    HW2   32
    5SOL     OW   33
    5SOL    HW1   34
    5SOL    HW2   35
    6SOL     OW   36
    6SOL    HW1   37
    6SOL    HW2   38
    7SOL     OW   39
    7SOL    HW1   40
    7SOL    HW2   41
    8SOL     OW   42
    8SOL    HW1   43
    8SOL    HW2   44
""".split('\n')

_COORDINATES_STR = """\
  -0.204   0.447   0.725
  -0.109   0.340   0.685
   0.032   0.401   0.675
   0.047   0.521   0.659
  -0.149   0.283   0.549
  -0.199   0.535   0.683
  -0.109   0.262   0.759
  -0.226   0.346   0.505
  -0.063   0.281   0.483
  -0.187   0.183   0.561
   0.134   0.321   0.686
   0.273   0.377   0.678
   0.364   0.277   0.607
   0.384   0.167   0.654
   0.325   0.402   0.819
   0.462   0.471   0.812
   0.226   0.491   0.895
   0.121   0.226   0.700
   0.271   0.470   0.623
   0.335   0.308   0.871
   0.455   0.557   0.747
   0.491   0.503   0.911
   0.535   0.401   0.773
   0.178   0.559   0.825
   0.151   0.430   0.942
   0.279   0.549   0.969
   0.421   0.315   0.496
   0.512   0.221   0.424
   0.657   0.257   0.455
   0.718   0.339   0.389
   0.488   0.230   0.274
   0.336   0.223   0.248
   0.562   0.114   0.206
   0.303   0.125   0.135
   0.404   0.405   0.460
   0.492   0.120   0.458
   0.527   0.324   0.237
   0.286   0.191   0.338
   0.301   0.321   0.221
   0.658   0.101   0.253
   0.504   0.024   0.216
   0.576   0.137   0.101
   0.378   0.133   0.057
   0.304   0.024   0.174\
"""
COORDINATES = [[float(x) for x in line.split()]
               for line in _COORDINATES_STR.split('\n')]
VELOCITIES = [[x + 1 for x in line] for line in COORDINATES]

DIFFERENCE_USE_CASE = (
    '', [], 'a', 2, 3, True, False, None, float('nan'), float('inf'),
    [True], [False], [True, False], [False, False], [None], [None, None],
    [1.1, 1.1], [0.0, 1.1],
    [float('nan'), float('inf')],
    [float('inf'), float('nan')],
    ['a', 'b'], 'ab',
    np.array([1.2, 2.3, 4.5]),
    np.array([[1.2, 2.3, 4.5], [5.6, 7.8, 8.9]]),
    np.array([np.nan, np.inf, 3]),
    [[6, 8], [9, 10, 11]],
    {'a': 3, 2: 'tata'},
    {'a': 3, 2: np.array([2, 3, 4])},
    {'a': 4, 2: 'not tata'},
)


@pytest.fixture(params=[True, False])
def gro_reference(request, tmpdir_factory):
    """
    Generate a GRO file and the corresponding molecule.
    """
    filename = tmpdir_factory.mktemp("data").join("tmp.gro")
    with open(str(filename), 'w') as outfile:
        write_ref_gro(outfile, velocities=request.param, box='10.0 11.1 12.2')
    molecule = build_ref_molecule(velocities=request.param)
    return filename, molecule


@pytest.fixture(params=[43, 45])
def gro_wrong_length(request, gro_reference, tmpdir_factory):  # pylint: disable=redefined-outer-name
    """
    Generate a GRO file with a wrong number of atoms on line 2.
    """
    path_in, _ = gro_reference
    path_out = tmpdir_factory.mktemp("data").join("wrong.gro")
    with open(str(path_in)) as infile, open(str(path_out), 'w') as outfile:
        outfile.write(next(infile))
        outfile.write('{}\n'.format(request.param))
        for line in infile:
            outfile.write(line)
    return path_out


def write_ref_gro(outfile, velocities=False, box='10.0 10.0 10.0'):
    """
    Write a GRO file from the reference data.

    Parameters
    ----------
    outfile: file
        The file in which to write.
    velocities: bool
        Set if velocities must be written.
    box: str
        The box string to write at the end of the file.
    """
    velocity_fmt = ''
    if velocities:
        velocity_fmt = '{:8.4f}' * 3
    outfile.write('Just a title\n')
    outfile.write(str(len(COORDINATES)) + '\n')
    for atom, coords, vels in zip(GRO_CONTENT, COORDINATES, VELOCITIES):
        outfile.write(('{}{:8.3f}{:8.3f}{:8.3f}' + velocity_fmt + '\n')
                      .format(atom, *itertools.chain(coords, vels)))
    outfile.write(box)
    outfile.write('\n')


def build_ref_molecule(velocities=False):
    """
    Build the molecule graph corresponding to the reference data.

    Parameters
    ----------
    velocities: bool
        Set if velocities must be written.

    Return
    ------
    Molecule
    """
    nodes = (
        {'resid': 1, 'resname': 'ALA', 'atomname': 'N', 'atomid': 1, 'element': 'N'},
        {'resid': 1, 'resname': 'ALA', 'atomname': 'CA', 'atomid': 2, 'element': 'C'},
        {'resid': 1, 'resname': 'ALA', 'atomname': 'C', 'atomid': 3, 'element': 'C'},
        {'resid': 1, 'resname': 'ALA', 'atomname': 'O', 'atomid': 4, 'element': 'O'},
        {'resid': 1, 'resname': 'ALA', 'atomname': 'CB', 'atomid': 5, 'element': 'C'},
        {'resid': 1, 'resname': 'ALA', 'atomname': 'H', 'atomid': 6, 'element': 'H'},
        {'resid': 1, 'resname': 'ALA', 'atomname': 'HA', 'atomid': 7, 'element': 'H'},
        {'resid': 1, 'resname': 'ALA', 'atomname': 'HB1', 'atomid': 8, 'element': 'H'},
        {'resid': 1, 'resname': 'ALA', 'atomname': 'HB2', 'atomid': 9, 'element': 'H'},
        {'resid': 1, 'resname': 'ALA', 'atomname': 'HB3', 'atomid': 10, 'element': 'H'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'N', 'atomid': 11, 'element': 'N'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'CA', 'atomid': 12, 'element': 'C'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'C', 'atomid': 13, 'element': 'C'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'O', 'atomid': 14, 'element': 'O'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'CB', 'atomid': 15, 'element': 'C'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'CG1', 'atomid': 16, 'element': 'C'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'CG2', 'atomid': 17, 'element': 'C'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'H', 'atomid': 18, 'element': 'H'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'HA', 'atomid': 19, 'element': 'H'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'HB', 'atomid': 20, 'element': 'H'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'HG11', 'atomid': 21, 'element': 'H'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'HG12', 'atomid': 22, 'element': 'H'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'HG13', 'atomid': 23, 'element': 'H'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'HG21', 'atomid': 24, 'element': 'H'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'HG22', 'atomid': 25, 'element': 'H'},
        {'resid': 2, 'resname': 'VAL', 'atomname': 'HG23', 'atomid': 26, 'element': 'H'},
        {'resid': 3, 'resname': 'SOL', 'atomname': 'OW', 'atomid': 27, 'element': 'O'},
        {'resid': 3, 'resname': 'SOL', 'atomname': 'HW1', 'atomid': 28, 'element': 'H'},
        {'resid': 3, 'resname': 'SOL', 'atomname': 'HW2', 'atomid': 29, 'element': 'H'},
        {'resid': 4, 'resname': 'SOL', 'atomname': 'OW', 'atomid': 30, 'element': 'O'},
        {'resid': 4, 'resname': 'SOL', 'atomname': 'HW1', 'atomid': 31, 'element': 'H'},
        {'resid': 4, 'resname': 'SOL', 'atomname': 'HW2', 'atomid': 32, 'element': 'H'},
        {'resid': 5, 'resname': 'SOL', 'atomname': 'OW', 'atomid': 33, 'element': 'O'},
        {'resid': 5, 'resname': 'SOL', 'atomname': 'HW1', 'atomid': 34, 'element': 'H'},
        {'resid': 5, 'resname': 'SOL', 'atomname': 'HW2', 'atomid': 35, 'element': 'H'},
        {'resid': 6, 'resname': 'SOL', 'atomname': 'OW', 'atomid': 36, 'element': 'O'},
        {'resid': 6, 'resname': 'SOL', 'atomname': 'HW1', 'atomid': 37, 'element': 'H'},
        {'resid': 6, 'resname': 'SOL', 'atomname': 'HW2', 'atomid': 38, 'element': 'H'},
        {'resid': 7, 'resname': 'SOL', 'atomname': 'OW', 'atomid': 39, 'element': 'O'},
        {'resid': 7, 'resname': 'SOL', 'atomname': 'HW1', 'atomid': 40, 'element': 'H'},
        {'resid': 7, 'resname': 'SOL', 'atomname': 'HW2', 'atomid': 41, 'element': 'H'},
        {'resid': 8, 'resname': 'SOL', 'atomname': 'OW', 'atomid': 42, 'element': 'O'},
        {'resid': 8, 'resname': 'SOL', 'atomname': 'HW1', 'atomid': 43, 'element': 'H'},
        {'resid': 8, 'resname': 'SOL', 'atomname': 'HW2', 'atomid': 44, 'element': 'H'},
    )
    for node, coords, vels in zip(nodes, COORDINATES, VELOCITIES):
        node['chain'] = ''
        node['position'] = np.array(coords)
        if velocities:
            node['velocity'] = np.array(vels)
    molecule = Molecule()
    molecule.add_nodes_from(enumerate(nodes))
    return molecule


def filter_molecule(molecule, exclude, ignh):
    """
    Remove nodes from a graph based on some criteria.

    Nodes are removed if their resname matches one from
    the `exclude` argument, or of `ighn` is `True` and the node is a
    hydrogen.

    The atoms are renumbered after the filtered atoms are removed. This will
    scramble the edges and the interactions! This function is meant to be used
    on molecules with ONLY nodes.

    Parameters
    ----------
    molecule: nx.Graph
        The molecule to filter. The molecule is modified in place.
    exclude: list
        List of residue name to exclude.
    ighn: bool
        If `True`, the hydrogens are excluded.
    """
    to_remove = []
    exclusions = [('resname', value) for value in exclude]
    if ignh:
        exclusions.append(('element', 'H'))
    for node_key, node in molecule.nodes.items():
        for key, value in exclusions:
            if key in node and node[key] == value:
                to_remove.append(node_key)
                break
    molecule.remove_nodes_from(to_remove)
    mapping = collections.OrderedDict(
        [(old, new) for new, old in enumerate(molecule.nodes)]
    )
    nx.relabel_nodes(molecule, mapping, copy=False)
    # Networkx mangles the order of the nodes, but we want the nodes sorted.
    new_nodes = sorted(molecule.nodes.items())
    molecule.clear()  # Remove all the nodes
    molecule.add_nodes_from(new_nodes)


def compare_dicts(testee, reference):
    """
    Report differences between dictionaries.
    """
    report = []
    dict_equals = testee.keys() == reference.keys()
    diff_keys = []
    if dict_equals:
        for key in testee:
            if are_different(testee[key], reference[key]):
                diff_keys.append(key)
    if diff_keys:
        report.append('The following keys differ: {}'.format(diff_keys))
    if diff_keys or not dict_equals:
        report.append('+ {}'.format(testee))
        report.append('- {}'.format(reference))
    return report


def assert_molecule_equal(molecule, reference):
    """
    Make a pytest test fail with a report if the molecules are not equal.
    """
    report = []
    # Nodes
    if len(molecule.nodes) != len(reference.nodes):
        report.append(
            'Different number of nodes: {} != {}'
            .format(len(molecule.nodes), len(reference.nodes))
        )
    zipped_nodes = zip(molecule.nodes.items(), reference.nodes.items())
    for (key_mol, node_mol), (key_ref, node_ref) in zipped_nodes:
        partial_report = []
        failed = False
        if key_mol != key_ref:
            partial_report.append('Node key {} != {}'.format(key_mol, key_ref))
            failed = True
        partial_report = compare_dicts(node_mol, node_ref)
        if partial_report:
            if not failed:
                report.append('Node {}'.format(key_mol))
            report.extend(partial_report)
        elif failed:
            report.append('= {}'.format(node_mol))

    extra_molecule_nodes = molecule.nodes.keys() - reference.nodes.keys()
    if extra_molecule_nodes:
        report.append('The following nodes are not in the reference: {}.'
                      .format(extra_molecule_nodes))
    extra_reference_nodes = reference.nodes.keys() - molecule.nodes.keys()
    if extra_reference_nodes:
        report.append('The following nodes are not in the molecule: {}.'
                      .format(extra_reference_nodes))
    # Edges
    # The order of the edges does not matter.
    molecule_edges = set(molecule.edges)
    reference_edges = set(reference.edges)
    extra_molecule_edges = molecule_edges - reference_edges
    extra_reference_edges = reference_edges - molecule_edges
    if extra_molecule_edges:
        report.append('The following edges are not in the reference:')
        report.append('{}'.format(extra_molecule_edges))
    if extra_reference_edges:
        report.append('The following edges are not in the molecule:')
        report.append('{}'.format(extra_reference_edges))
    # Meta data
    report.extend(compare_dicts(molecule.meta, reference.meta))
    if molecule._force_field is not reference._force_field:  # pylint: disable=protected-access
        report.append('Force fields do not match: {} != {}'
                      .format(molecule._force_field, reference._force_field))  # pylint: disable=protected-access
    if molecule.nrexcl != reference.nrexcl:
        report.append('nrexcl: {} != {}'.format(molecule.nrexcl, reference.nrexcl))
    # Interactions
    if report:
        pytest.fail('\n'.join(report))


@st.composite
def generate_dict(draw, min_size=0):
    """
    Strategy to generate an arbitrary dictionary.
    """
    keys = st.one_of(st.text(), st.integers())
    values = st.one_of(
        st.text(), st.integers(), st.floats(),
        hnp.arrays(dtype=st.one_of(hnp.integer_dtypes(), hnp.floating_dtypes(), hnp.complex_number_dtypes()), shape=hnp.array_shapes())
    )
    dict_a = draw(st.dictionaries(keys, values, min_size=min_size, max_size=10))
    return dict_a


@st.composite
def generate_equal_dict(draw):
    """
    Strategy to generate two equal dictionaries.
    """
    dict_a = draw(generate_dict())
    dict_b = copy.deepcopy(dict_a)
    return dict_a, dict_b



@st.composite
def generate_diff_dict(draw):
    """
    Strategy to generate to similar but different dictionaries.
    """
    dict_a = draw(generate_dict(min_size=1))
    dict_b = copy.deepcopy(dict_a)
    values = st.one_of(
        st.text(), st.integers(), st.floats(),
        hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes())
    )
    num_to_change = draw(st.integers(
        min_value=1, max_value=(len(dict_b) // 2) + 1
    ))
    for _ in range(num_to_change):
        key = draw(st.sampled_from(list(dict_b.keys())))
        new_val = draw(values)
        assume(are_different(new_val, dict_a[key]))
        dict_b[key] = new_val
    dict_c = draw(generate_dict())
    for key in set(dict_c.keys()) - set(dict_b.keys()):
        dict_b[key] = dict_c[key]
    return dict_a, dict_b


@given(generate_equal_dict())
def test_compare_dict_equal(dict_a_and_b):
    """
    Test that :func:`compare_dicts` identify equal dictionaries.
    """
    dict_a = dict_a_and_b[0]
    dict_b = dict_a_and_b[1]
    assert not compare_dicts(dict_a, dict_b)  # report is empty


@given(generate_diff_dict())
def test_compare_dict_diff(dict_a_and_b):
    """
    Test that :func:`compare_dicts` identify different dictionaries.
    """
    dict_a = dict_a_and_b[0]
    dict_b = dict_a_and_b[1]
    assert compare_dicts(dict_a, dict_b)  # report is not empty


def test_filter_molecule_none(gro_reference):  # pylint: disable=redefined-outer-name
    """
    Test that :func:`filter_molecule` does not remove nodes when not asked for.
    """
    _, molecule = gro_reference
    filter_molecule(molecule, exclude=(), ignh=False)
    assert len(molecule.nodes) == 44


def test_filter_molecule_ignh(gro_reference):  # pylint: disable=redefined-outer-name
    """
    Test that :func:`filter_molecule` works with `ighn` argument.
    """
    _, molecule = gro_reference
    filter_molecule(molecule, exclude=(), ignh=True)
    elements = [node['element'] for node in molecule.nodes.values()]
    assert len(molecule.nodes) == (44 - 26)
    assert 'H' not in elements


@pytest.mark.parametrize('exclude', (
    ('SOL', ), ('ALA', ), ('ALA', 'VAL'),
    ('ALA', 'VAL', 'SOL'), ('XXX', ), ('ALA', 'XXX'),
))
def test_filter_molecule_exclude(gro_reference, exclude):  # pylint: disable=redefined-outer-name
    """
    Test that :func:`filter_molecule` works with `exclude` argument.
    """
    _, molecule = gro_reference
    filter_molecule(molecule, exclude=exclude, ignh=False)
    resnames = [node['resname'] for node in molecule.nodes.values()]
    should_not_be_there = [exclusion
                           for exclusion in exclude
                           if exclusion in resnames]
    assert not should_not_be_there


def test_filter_molecule_identity(gro_reference):  # pylint: disable=redefined-outer-name
    """
    Test that :func:`filter_molecule` modifies the molecule in place.
    """
    _, molecule = gro_reference
    id_before = id(molecule)
    filter_molecule(molecule, exclude=('SOL', ), ignh=True)
    assert id(molecule) == id_before


def test_filter_molecule_order(gro_reference):  # pylint: disable=redefined-outer-name
    """
    Test that :func:`filter_molecule` returns the nodes in order.
    """
    _, molecule = gro_reference
    filter_molecule(molecule, exclude=('SOL', ), ignh=True)
    keys = list(molecule.nodes)
    assert  sorted(keys) == keys


@pytest.mark.parametrize('exclude', (
    (), ('SOL', ), ('ALA', ), ('ALA', 'VAL'),
    ('ALA', 'VAL', 'SOL'), ('XXX', ), ('ALA', 'XXX'),
))
@pytest.mark.parametrize('ignh', (True, False))
def test_read_gro(gro_reference, exclude, ignh):  # pylint: disable=redefined-outer-name
    """
    Test the GRO reader.
    """
    filename, reference = gro_reference
    filter_molecule(reference, exclude=exclude, ignh=ignh)
    molecule = gro.read_gro(filename, exclude=exclude, ignh=ignh)
    pprint(list(molecule.nodes.items()))
    assert_molecule_equal(molecule, reference)


def test_read_gro_wrong_atom_number(gro_wrong_length):  # pylint: disable=redefined-outer-name
    """
    Test that the GRO reader raises an exception if the number of atoms is not
    consistent.
    """
    with pytest.raises(ValueError):
        gro.read_gro(gro_wrong_length)


def test_write_gro(gro_reference, tmpdir):
    """
    Test writing GRO file.
    """
    filename, molecule = gro_reference
    system = vermouth.System()
    system.molecules.append(molecule)
    outname = tmpdir / 'out_test.gro'
    gro.write_gro(
        system,
        outname,
        box=(10.0, 11.1, 12.2),
        title='Just a title',
    )
    DeferredFileWriter().write()
    with open(str(filename)) as ref, open(str(outname)) as out:
        assert out.read() == ref.read()
