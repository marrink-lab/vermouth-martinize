# Copyright 2024 University of Groningen
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

from ..processors.processor import Processor
import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.spatial import cKDTree as KDTree
from .. import MergeAllMolecules
from ..graph_utils import make_residue_graph


PROTEIN_MAP = {
    "ALA": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0}
    },
    "ARG": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.88, 'atype': 7},
        'NE':       {'vrad': 1.64, 'atype': 3},
        'CZ':       {'vrad': 1.61, 'atype': 6},
        'NH1':      {'vrad': 1.64, 'atype': 3},
        'NH2':      {'vrad': 1.64, 'atype': 3},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0}
    },
    "ASN": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 6},
        'OD1':      {'vrad': 1.42, 'atype': 2},
        'ND2':      {'vrad': 1.64, 'atype': 3},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "ASP": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 6},
        'OD1':      {'vrad': 1.46, 'atype': 2},
        'OD2':      {'vrad': 1.42, 'atype': 2},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "CYM": {},
    "CYX": {},
    "CYS": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'SG':       {'vrad': 1.77, 'atype': 6},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "GLN": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.61, 'atype': 6},
        'OE1':      {'vrad': 1.42, 'atype': 2},
        'NE2':      {'vrad': 1.64, 'atype': 3},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "GLU": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.61, 'atype': 6},
        'OE1':      {'vrad': 1.46, 'atype': 2},
        'OE2':      {'vrad': 1.42, 'atype': 2},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "GLY": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':        {'vrad': 1.88, 'atype': 6},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "HIE": {},
    "HIP": {},
    "HIS": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 5},
        'ND1':      {'vrad': 1.64, 'atype': 1},
        'CD2':      {'vrad': 1.76, 'atype': 5},
        'CE1':      {'vrad': 1.76, 'atype': 5},
        'NE2':      {'vrad': 1.64, 'atype': 1},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "ILE": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG1':      {'vrad': 1.88, 'atype': 4},
        'CG2':      {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "LEU": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD1':      {'vrad': 1.88, 'atype': 4},
        'CD2':      {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "LYS": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.88, 'atype': 4},
        'CE':       {'vrad': 1.88, 'atype': 7},
        'NZ':       {'vrad': 1.64, 'atype': 3},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "MET": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'SD':       {'vrad': 1.77, 'atype': 8},
        'CE':       {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "PHE": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 5},
        'CD1':      {'vrad': 1.61, 'atype': 5},
        'CD2':      {'vrad': 1.76, 'atype': 5},
        'CE1':      {'vrad': 1.76, 'atype': 5},
        'CE2':      {'vrad': 1.76, 'atype': 5},
        'CZ':       {'vrad': 1.76, 'atype': 5},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "PRO": {
        'N':        {'vrad': 1.64, 'atype': 6},
        'CA':       {'vrad': 1.88, 'atype': 4},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "SER": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 6},
        'OG':       {'vrad': 1.46, 'atype': 1},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "THR": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 6},
        'OG1':      {'vrad': 1.46, 'atype': 1},
        'CG2':      {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "TRP": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 5},
        'CD1':      {'vrad': 1.76, 'atype': 5},
        'CD2':      {'vrad': 1.61, 'atype': 5},
        'NE1':      {'vrad': 1.64, 'atype': 3},
        'CE2':      {'vrad': 1.61, 'atype': 5},
        'CE3':      {'vrad': 1.76, 'atype': 5},
        'CZ2':      {'vrad': 1.76, 'atype': 5},
        'CZ3':      {'vrad': 1.76, 'atype': 5},
        'CH2':      {'vrad': 1.76, 'atype': 5},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "TYR": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 5},
        'CD1':      {'vrad': 1.76, 'atype': 5},
        'CD2':      {'vrad': 1.76, 'atype': 5},
        'CE1':      {'vrad': 1.76, 'atype': 5},
        'CE2':      {'vrad': 1.76, 'atype': 5},
        'CZ':       {'vrad': 1.61, 'atype': 5},
        'OH':       {'vrad': 1.46, 'atype': 1},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "VAL": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG1':      {'vrad': 1.88, 'atype': 4},
        'CG2':      {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    }
}


def get_vdw_radius(resname, atomname):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    res_vdw = PROTEIN_MAP[resname]
    try:
        atom_vdw = res_vdw[atomname]
    except KeyError:
        atom_vdw = res_vdw['default']
    return atom_vdw['vrad']


def get_atype(resname, atomname):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    res_vdw = PROTEIN_MAP[resname]
    try:
        atom_vdw = res_vdw[atomname]
    except KeyError:
        atom_vdw = res_vdw['default']
    return atom_vdw['atype']


def make_surface(position, fiba, fibb, vrad):
    """
    Generate points on a sphere using Fibonacci points

    position: np.array
        shape (3,) array of an atomic position to build a sphere around
    fiba: int. n-1 fibonacci number to build number of points on sphere
    fibb: int. n fibonacci number to build number of points on sphere
    vrad: float. VdW radius of the input atom to build a sphere around.

    position: centre of sphere
    """
    x, y, z = position
    phi_aux = 0

    surface = np.zeros((0, 3))
    for k in range(fibb):

        phi_aux += fiba
        if phi_aux > fibb:
            phi_aux -= fibb

        theta = np.arccos(1.0 - 2.0 * k / fibb)
        phi = 2.0 * np.pi * phi_aux / fibb
        surface_x = x + vrad * np.sin(theta) * np.cos(phi)
        surface_y = y + vrad * np.sin(theta) * np.sin(phi)
        surface_z = z + vrad * np.cos(theta)
        surface = np.vstack((surface, np.array([surface_x, surface_y, surface_z])))
    return surface


def atom2res(arrin, residues, nresidues, norm=False):
    """
    take an array with atom level data and sum the entries over within the residue

    arrin: np.ndarray
        NxN array of entries for each atom
    residues: np.array
        array of length N indicating which residue an atom belongs to
    nresidues: int
        number of residues in the molecule
    norm: bool
        if True, then any entry > 0 in the summed array = 1
    """
    out = np.array([int(arrin[np.where(residues == i)[0], np.where(residues == j)[0][:, np.newaxis]].sum())
                    for i in range(nresidues)
                    for j in range(nresidues)]).reshape((nresidues, nresidues))
    if norm:
        out[out > 0] = 1

    return out


def bondtype(i, j):
    maxatomtype = 10
    assert 1 <= i <= maxatomtype
    assert 1 <= j <= maxatomtype

    i -= 1
    j -= 1
    # BOND TYPE
    # Types of contacts:
    # HB -- 1 -- hydrogen-bond
    # PH -- 2 -- hydrophobic
    # AR -- 3 -- aromatic - contacts between aromatic rings
    # IB -- 4 -- ionic bridge - contacts created by two atoms with different charges
    # DC -- 5 -- destabilizing contact - contacts which are in general repulsive
    # OT -- 6 -- denotes negligible other contacts.
    # 1-HB,2-PH,3-AR,4-IP,5-DC,6-OT
    btype = np.array([[1, 1, 1, 5, 5, 6, 6, 6, 1, 1],
                      [1, 5, 1, 5, 5, 6, 6, 6, 1, 5],
                      [1, 1, 5, 5, 5, 6, 6, 6, 5, 1],
                      [5, 5, 5, 2, 2, 6, 6, 6, 5, 5],
                      [5, 5, 5, 2, 3, 6, 6, 6, 5, 5],
                      [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [1, 1, 5, 5, 5, 6, 6, 6, 5, 4],
                      [1, 5, 1, 5, 5, 6, 6, 6, 4, 5]])
    return btype[i][j]


def contact_info(system):

    system = MergeAllMolecules().run_system(system)
    G = make_residue_graph(system.molecules[0])

    resids = []
    chains = []
    resnames = []
    positions_all = []
    ca_pos = []
    vdw_list = []
    atypes = []
    res_serial = []
    nodes = []
    for residue in G.nodes:
        # we only need these for writing at the end
        resnames.append(G.nodes[residue]['resname'])
        resids.append(G.nodes[residue]['resid'])
        chains.append(G.nodes[residue]['chain'])
        nodes.append(G.nodes[residue]['_res_serial'])

        for atom in sorted(G.nodes[residue]['graph'].nodes):
            if 'position' in G.nodes[residue]['graph'].nodes[atom]:
                res_serial.append(G.nodes[residue]['graph'].nodes[atom]['_res_serial'])

                positions_all.append(G.nodes[residue]['graph'].nodes[atom]['position'] * 10)

                vdw_list.append(get_vdw_radius(G.nodes[residue]['graph'].nodes[atom]['resname'],
                                               G.nodes[residue]['graph'].nodes[atom]['atomname']))
                atypes.append(get_atype(G.nodes[residue]['graph'].nodes[atom]['resname'],
                                        G.nodes[residue]['graph'].nodes[atom]['atomname']))

            if G.nodes[residue]['graph'].nodes[atom]['atomname'] == 'CA':
                ca_pos.append(G.nodes[residue]['graph'].nodes[atom]['position'])


    vdw_list = np.array(vdw_list)
    atypes = np.array(atypes)
    coords = np.stack(positions_all)
    res_serial = np.array(res_serial)

    resids = np.array(resids)
    chains = np.array(chains)
    resnames = np.array(resnames)
    nodes = np.array(nodes)

    # 2) find the number of residues that we have
    nresidues = len(G)

    return vdw_list, atypes, coords, res_serial, resids, chains, resnames, nodes, ca_pos, nresidues, G


def calculate_contact_map(vdw_list, atypes, coords, res_serial,
                          resids, chains, resnames, nodes, ca_pos, nresidues,
                          G):

    # some initial definitions of variables that we need
    fib = 14
    fiba, fibb = 0, 1
    for _ in range(fib):
        fiba, fibb = fibb, fiba + fibb
    natoms = len(coords)

    alpha = 1.24  # Enlargement factor for attraction effects
    water_radius = 2.80  # Radius of a water molecule in A

    coords_tree = KDTree(coords)
    # all_vdw = np.array([x for xs in
    #                     [[PROTEIN_MAP[i][j]['vrad'] for j in PROTEIN_MAP[i].keys()]
    #                      for i in PROTEIN_MAP.keys()]
    #                     for x in xs])
    over = np.zeros((len(coords), len(coords)))
    vdw_max = 1.88 # all_vdw.max()
    over_sdm = coords_tree.sparse_distance_matrix(coords_tree, 2 * vdw_max * alpha)
    for (idx, jdx), distance_between in over_sdm.items():
        if idx != jdx:
            if distance_between < (vdw_list[idx] + vdw_list[jdx]) * alpha:
                over[idx, jdx] = 1

    # set up the surface overlap criterion
    # generate fibonacci spheres for all atoms.
    # can't decide whether quicker/better for memory to generate all in one go here
    # or incorporate into the loop. code to do it more on the fly is left in the loop for now
    spheres = np.stack([KDTree(make_surface(i, fiba, fibb, water_radius + j))
                        for i, j in zip(coords, vdw_list)])

    hit_results = np.full((natoms, fibb), -1)
    dists_counter = np.full((natoms, fibb), np.inf)
    surface_sdm = coords_tree.sparse_distance_matrix(coords_tree, (2 * vdw_max) + water_radius)
    for (idx, jdx), distance_between in surface_sdm.items():
        if idx != jdx:

            if distance_between < (vdw_list[idx] + vdw_list[jdx] + water_radius):

                base_tree = spheres[idx]
                vdw = vdw_list[jdx] + water_radius

                res = np.array(base_tree.query_ball_point(coords[jdx], vdw))
                if len(res) > 0:
                    to_fill = np.where(distance_between < dists_counter[idx][res])[0]

                    dists_counter[idx][res[to_fill]] = distance_between
                    hit_results[idx][res[to_fill]] = jdx

    contactcounter_1 = np.zeros((natoms, natoms))
    stabilisercounter_1 = np.zeros((natoms, natoms))
    destabilisercounter_1 = np.zeros((natoms, natoms))

    for i, j in enumerate(hit_results):
        for k in j:
            if k >= 0:
                at1 = atypes[i]
                at2 = atypes[k]
                if (at1 > 0) and (at2 > 0):
                    contactcounter_1[i, k] += 1
                    btype = bondtype(at1, at2)
                    if btype <= 4:
                        stabilisercounter_1[i, k] += 1
                    if btype == 5:
                        destabilisercounter_1[i, k] += 1

    overlapcounter_2 = atom2res(over, res_serial, nresidues, norm=True)
    contactcounter_2 = atom2res(contactcounter_1, res_serial, nresidues)
    stabilisercounter_2 = atom2res(stabilisercounter_1, res_serial, nresidues)
    destabilisercounter_2 = atom2res(destabilisercounter_1, res_serial, nresidues)

    # # this to write out the file if needed
    # with open('contact_map_vermouth.out', 'w') as f:
    #     count = 0
    #     for i1 in range(nresidues):
    #         for i2 in range(nresidues):
    #             over = overlapcounter_2[i1, i2]
    #             cont = contactcounter_2[i1, i2]
    #             stab = stabilisercounter_2[i1, i2]
    #             dest = destabilisercounter_2[i1, i2]
    #             ocsu = stab
    #             rcsu = stab - dest
    #
    #             if (over > 0 or cont > 0) and (i1 != i2):
    #                 a = np.where(nodes == i1)[0][0]
    #                 b = np.where(nodes == i2)[0][0]
    #                 count += 1
    #                 msg = (f"R {int(count):6d} "
    #                        f"{int(i1 + 1):5d}  {G.nodes[a]['resname']:3s} {G.nodes[a]['chain']:1s} {int(G.nodes[a]['resid']):4d}    "
    #                        f"{int(i2 + 1):5d}  {G.nodes[b]['resname']:3s} {G.nodes[b]['chain']:1s} {int(G.nodes[b]['resid']):4d}    "
    #                        f"{euclidean(ca_pos[a], ca_pos[b])*10:9.4f}     "
    #                        f"{int(over):1d} {1 if cont != 0 else 0} {1 if ocsu != 0 else 0} {1 if rcsu > 0 else 0}"
    #                        f"{int(rcsu):6d}  {int(cont):6d}\n")
    #                 f.writelines(msg)

    contacts = []
    for i1 in range(nresidues):
        for i2 in range(nresidues):
            over = overlapcounter_2[i1, i2]
            cont = contactcounter_2[i1, i2]
            stab = stabilisercounter_2[i1, i2]
            dest = destabilisercounter_2[i1, i2]
            # ocsu = stab
            rcsu = 1 if (stab - dest) > 0 else 0

            if (over > 0 or cont > 0) and (i1 != i2):
                a = np.where(nodes == i1)[0][0]
                b = np.where(nodes == i2)[0][0]
                if over == 1 or (over == 0 and rcsu == 1):
                    # this is a OV or rCSU contact we take it
                    contacts.append((int(G.nodes[a]['resid']), G.nodes[a]['chain'],
                                     int(G.nodes[b]['resid']), G.nodes[b]['chain']))

    return contacts


"""
Read RCSU Go model contact maps.
"""


def read_go_map(file_path):
    """
    Read a RCSU contact map from the c code as published in
    doi:10.5281/zenodo.3817447. The format requires all 
    contacts to have 18 columns and the first column to be 
    a capital R.

    Parameters
    ----------
    file_path: :class:`pathlib.Path`
        path to the contact map file

    Returns
    -------
    list(tuple)
        contact as chain id, res id, chain id, res id
    """
    with open(file_path, "r", encoding='UTF-8') as _file:
        contacts = []
        for line in _file:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue

            if tokens[0] == "R" and len(tokens) == 18:
                # this is a bad place to filter but follows
                # the old script
                if tokens[11] == "1" or (tokens[11] == "0" and tokens[14] == "1"):
                    # this is a OV or rCSU contact we take it
                    contacts.append((int(tokens[5]), tokens[4], int(tokens[9]), tokens[8]))

        if len(contacts) == 0:
            raise IOError("You contact map is empty. Are you sure it has the right formatting?")
    return contacts


class GenerateContactMap(Processor):
    """
    Processor to generate the contact rCSU contact map for a protein from an atomistic structure
    """
    def __init__(self, path=None):
        self.path = path

    def run_system(self, system):
        """
        Process `system`.

        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        self.system = system

        if self.path is None:
            vdw_list, atypes, coords, res_serial, resids, chains, resnames, nodes, ca_pos, nresidues, G = contact_info(
                system)
            self.system.go_params["go_map"].append(calculate_contact_map(vdw_list,
                                                                         atypes,
                                                                         coords,
                                                                         res_serial,
                                                                         resids,
                                                                         chains,
                                                                         resnames,
                                                                         nodes,
                                                                         ca_pos,
                                                                         nresidues,
                                                                         G))
        else:
            self.system.go_params["go_map"].append(read_go_map(file_path=self.path))
