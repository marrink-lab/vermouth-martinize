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

@settings(deadline=None, max_examples=50)
@given(st.data())
def test_constrained_kmeans(data):

    dims = st.integers(min_value=3, max_value=3)
    num_points = st.integers(min_value=1, max_value=15)
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
    print(len(system.molecules))
    MoleculeGrouper(init_clusters='fixed', size_tries=3, clust_sizes=3, tolerances=1).run_system(system)
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
    assert np.allclose(sorted(anisotropies), [
        0.10651194938585429, 0.15115950311614146, 0.16819992894453728,
        0.1951616957409924, 0.19855980389681138, 0.22321389514533896,
        0.25737841997945143, 0.2634401201510208, 0.27199936926291157,
        0.27317572289777503, 0.27366830952828114, 0.27497966687565445,
        0.27570889176866265, 0.27963397440123494, 0.282086428114358,
        0.28642105844714805, 0.28696926120555066, 0.28718744100113336,
        0.28768513658428263, 0.28875695534318735, 0.2899822837196525,
        0.30078916919045373, 0.30081991679605313, 0.30267057595162306,
        0.3028515691099407, 0.30287393731736556, 0.3051870785165556,
        0.3161546531966819, 0.31667845841706066, 0.31855574564490496,
        0.31975887891878596, 0.32059399905763275, 0.3227298659848604,
        0.32336061044202213, 0.32700457113318504, 0.329983108839937,
        0.3396497253975489, 0.343405078110529, 0.3528181819994419,
        0.3573803627725072, 0.35991150525533255, 0.3734480075272171,
        0.377372116577117, 0.37774419291866257, 0.3874537145473449,
        0.39315341495350953, 0.39346767732251664, 0.3959414962233947,
        0.39858987842762805, 0.39939767420823635, 0.40314754118360874,
        0.40663026419327986, 0.40751887118452845, 0.4085492813049517,
        0.4138928537240878, 0.4161377352949154, 0.4247954154992609,
        0.4285259032891967, 0.43159648564764863, 0.43627564167147637,
        0.4470502826116086, 0.44991274434242323, 0.4582410382652621,
        0.459211944199475, 0.4623123004624836, 0.5024676409068336,
        0.5043412732049195, 0.5070192521911536, 0.5176758569227614,
        0.5197618555081338, 0.5236485289490618, 0.5575590333472291,
        0.5613578319207018, 0.5632033530585567, 0.5684642317962016,
        0.5688998238107379, 0.5742549677311761, 0.5808113556336375,
        0.5825656880157071, 0.5857515663968016, 0.588857322278671,
        0.5895388960599243, 0.5986528208538493, 0.5990525101004103,
        0.6016123889533482, 0.6027445822862898, 0.6033039216118167,
        0.6133149617543336, 0.6175561122006982, 0.6220104968414778,
        0.6254936046046898, 0.6391659990523493, 0.6392470738824115,
        0.6406042397148723, 0.6536865099589015, 0.6557568075652149,
        0.6818951795565646, 0.696640636031432, 0.7025547863403951,
        0.7066403635880708, 0.7174026268460711, 0.7216746030508401,
        0.7287062778088912, 0.7471205680988195, 0.7604323965290503,
        0.7673614721218632, 0.7738429600057989, 0.7742938083406593,
        0.7751245512292301, 0.7847558767885271, 0.7882386297307716,
        0.7985139784645476, 0.8134731437128435, 0.8264524922460024,
        0.8492966992667246, 0.8561955106767053, 0.8665228734540954,
        0.870468007836023, 0.8723614384660241, 0.8741594447031515,
        0.8930954821759747, 0.9075450154337484, 0.9098525125266149,
        0.9107435662088775, 0.9190125205380254, 0.9338142616567598,
        0.9415496023754201, 0.9442553681849799, 0.9498228784967794,
        0.9519102423917406, 0.9596116981230298, 0.9778335759938078,
        0.9832463082236982, 0.9843441417660237, 0.9859317922004184,
        0.9866622067104501, 0.9870552304526536, 0.9878940578188086,
        0.9882012003002396, 0.9884023207637785, 0.9895272420986398,
        0.9896729148454506, 0.9917544341504739, 0.9925751774216135,
        0.9931452287323225, 0.993792857314373, 0.995159253214102,
        0.9951761548589004, 0.9956363913601163, 0.9958575474555383,
        0.9962350875318009, 0.9975063971984528])
