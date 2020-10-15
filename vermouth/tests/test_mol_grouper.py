# Copyright 2020 University of Groningen
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

from .datafiles import PDB_1AQP, FF_UNIVERSAL_TEST

from hypothesis import given, settings
from hypothesis import strategies as st
import hypothesis.extra.numpy as npst

from functools import partial
import networkx as nx
import numpy as np
from vermouth.processors.molecule_grouper import constrained_kmeans, expand_to_list, MoleculeGrouper
from vermouth.pdb import read_pdb
from vermouth.processors.make_bonds import MakeBonds
from vermouth.system import System
from vermouth.forcefield import ForceField

def _is_valid_clust_tol(clust_spec, npoints, n_clusters):
    clust_size, tolerance = clust_spec
    clust_size = expand_to_list(clust_size, n_clusters)
    tolerance = expand_to_list(tolerance, n_clusters)

    low = 0
    high = 0
    for size, tol in zip(clust_size, tolerance):
        low += max(size - tol, 0)
        high += size + tol

    return low <= npoints <= high


def finite_real_arrays(shape):
    return npst.arrays(dtype=float,
                       elements=st.floats(min_value=-1e8, max_value=1e8,
                                          allow_nan=False, allow_infinity=False),
                       shape=shape)

@settings(deadline=None)
@given(st.data())
def test_constrained_kmeans(data):

    dims = st.integers(min_value=1, max_value=3)
    num_points = st.integers(min_value=1, max_value=50)
    point_shape = data.draw(st.tuples(num_points, dims), label='point_shape')
    points = finite_real_arrays(point_shape)
    n_points = point_shape[0]
    n_clusters = data.draw(st.integers(min_value=1, max_value=n_points), label='n_clusters')

    clust_sizes = st.one_of(st.integers(min_value=0, max_value=n_points),
                            st.lists(st.integers(min_value=0, max_value=n_points),
                                     min_size=n_clusters, max_size=n_clusters))
    tolerances = st.one_of(st.integers(min_value=0),
                           st.lists(st.integers(min_value=0),
                                    min_size=n_clusters, max_size=n_clusters))
    clust_specs = st.tuples(clust_sizes, tolerances).filter(
        partial(_is_valid_clust_tol, npoints=n_points, n_clusters=n_clusters)
    )

    inits = st.one_of(
            st.just('random'),
            st.just('fixed'),
            finite_real_arrays((n_clusters, point_shape[1]))
        )

    point, clust_spec, init = data.draw(st.tuples(points, clust_specs, inits))
    clust_size, tolerance = clust_spec

    cost, clusters, memberships, iter = constrained_kmeans(point, n_clusters,
                                                           clust_sizes=clust_size,
                                                           tolerances=tolerance,
                                                           init_clusters=init)
    assert memberships.sum() == n_points  # Every point gets assigned
    assert np.all(memberships.sum(axis=1) == 1)  # Every point gets assigned once
    assert clusters.shape == (n_clusters, point_shape[1])
    assert np.allclose(memberships.T.dot(point)/memberships.sum(axis=0, keepdims=True).T, clusters, equal_nan=True)
    assert np.all(memberships.sum(axis=0) <= np.array(clust_size) + tolerance)
    assert np.all(memberships.sum(axis=0) >= np.array(clust_size) - tolerance)


def test_mol_grouper_processor():
    molecules = read_pdb(PDB_1AQP, ignh=False,
                         exclude=['BNG', 'ALA', 'ARG', 'ASP', 'ASN', 'CYS',
                                  'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                                  'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                  'SER', 'THR', 'TRP', 'TYR', 'VAL'])
    ff = ForceField(FF_UNIVERSAL_TEST)
    system = System(force_field=ff)
    system.molecules.extend(molecules)
    MakeBonds(allow_dist=False).run_system(system)

    MoleculeGrouper().run_system(system)
    orig_atom_ids = set()
    for orig_mol in molecules:
        ids = set(nx.get_node_attributes(orig_mol, 'atomid').values())
        assert len(ids) == len(orig_mol)  # Defensive
        orig_atom_ids |= ids

    masses = {'H': 1, 'O': 16}
    anisotropies = []

    atom_ids = set()
    for mol in system.molecules:
        ids = set(nx.get_node_attributes(mol, 'atomid').values())
        assert len(ids) == len(mol)  # Defensive
        atom_ids |= ids

        mass_pos = [(masses[mol.nodes[idx]['element']], mol.nodes[idx]['position']) for idx in mol]
        mass, pos = map(np.array, zip(*mass_pos))

        com = np.average(pos, axis=0, weights=mass)
        rpos = pos - com
        # Gyration tensor
        gyr = 1/mass.sum() * (mass[:, None, None] * rpos[:, None, :] * rpos[:, :, None]).sum(axis=0)
        evals = np.sort(np.linalg.eigvals(gyr))
        anisotropy = 1 - 3*(evals[0]*evals[1] + evals[0]*evals[2] + evals[1]*evals[2])/evals.sum()**2
        # asphericity = evals[2]**2 - 0.5*(evals[0]**2 + evals[1]**2)
        anisotropies.append(anisotropy)

    print([len(mol) for mol in system.molecules])
    print(sorted(anisotropies))
    print(np.mean(anisotropies), np.median(anisotropies))
    print(np.min(anisotropies), np.max(anisotropies))
    assert atom_ids == orig_atom_ids  # Assert all atoms still exist
    assert sorted(anisotropies) == [
        0.25085593795965455, 0.2529047764915312, 0.25323958307578975,
        0.2537597454373204, 0.2548631381443849, 0.2573355058785781,
        0.26115181286902944, 0.26632113708996064, 0.2680643923601431,
        0.27038742416212236, 0.2704100372689785, 0.2849979804485867,
        0.28683062185716257, 0.29568241443373355, 0.3028840397545486,
        0.3036877588488638, 0.3194198616942183, 0.31963114871237086,
        0.34996692130183216, 0.35268395254118035, 0.35333076507269434,
        0.3571085960533166, 0.35910764114740923, 0.35932172944163265,
        0.3809949406791535, 0.400694539905994, 0.4241811974762486,
        0.43028022963998136, 0.4306327776345614, 0.4329990354556318,
        0.44096494593470736, 0.44470123526676486, 0.4512465910646718,
        0.4580749330554146, 0.4650605339997147, 0.4843010541986611,
        0.4851803578096423, 0.4944110333517201, 0.5090317943917886,
        0.5128890853319927, 0.514500877226407, 0.5211771287698562,
        0.5276872454818164, 0.5285788366472837, 0.5546071988890144,
        0.5633177084420308, 0.5675714092867107, 0.569556023910411,
        0.5863100645751278, 0.5949873238659845, 0.6095161096380238,
        0.6207848370689724, 0.6307498477422682, 0.6508769190216402,
        0.6754807347866015, 0.6810511840158268, 0.6823798589781285,
        0.6854311692445532, 0.689251848207091, 0.6896050647360608,
        0.6962196761751186, 0.7043194302230061, 0.7050020407845516,
        0.7052420335369236, 0.7119596123888705, 0.7184070054731,
        0.7232867086631718, 0.7297004554264764, 0.7354934064179939,
        0.7355596069337991, 0.744442209025438, 0.7486773231707785,
        0.7493370689585721, 0.7592319135261281, 0.7605041022202976,
        0.7639436509413724, 0.7744147912598952, 0.7805926408808075,
        0.783872128298106, 0.7843416139784609, 0.7993799846208003,
        0.8024700247753102, 0.820204624604423, 0.8234190191564019,
        0.8237005787527754, 0.8237796850185399, 0.8250612870558544,
        0.8279223528341052, 0.8280642251282214, 0.8370794889146358,
        0.8458438053764618, 0.8502608803765873, 0.8610727186183785,
        0.8711544635103854, 0.8743899737488474, 0.8752565777007273,
        0.8940768360867087, 0.8955786055123104, 0.8983032128817957,
        0.9001189382997966, 0.9028185300090261, 0.9152430766234718,
        0.9293826469543237, 0.9420764732850262, 0.9470722811723375,
        0.9486411944222164, 0.9548590140960376, 0.9714360338141241,
        0.9790887482462443, 0.9824275749046683, 0.9841247421399735,
        0.9926104368377939, 0.9952918987723255, 0.9983152951363736]
