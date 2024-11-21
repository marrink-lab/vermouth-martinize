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
# import numpy.ma as ma
import time

# REFERENCES FOR PROTEIN MAP
# REFERENCE: J. Tsai, R. Taylor, C. Chothia, and M. Gerstein, J. Mol. Biol 290:290 (1999)
# REFERENCE: https:# aip.scitation.org/doi/suppl/10.1063/1.4929599/suppl_file/sm.pdf
PROTEIN_MAP = {
    "ALA": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'default': {'vrad': 0.00, 'atype': 0}
    },
    "ARG": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA ': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.88, 'atype': 4},
        'CD': {'vrad': 1.88, 'atype': 7},
        'NE': {'vrad': 1.64, 'atype': 3},
        'CZ': {'vrad': 1.61, 'atype': 6},
        'NH1': {'vrad': 1.64, 'atype': 3},
        'NH2': {'vrad': 1.64, 'atype': 3},
        'default': {'vrad': 0.00, 'atype': 0}
    },
    "ASN": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.61, 'atype': 6},
        'OD1': {'vrad': 1.42, 'atype': 2},
        'ND2': {'vrad': 1.64, 'atype': 3},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "ASP": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.61, 'atype': 6},
        'OD1': {'vrad': 1.46, 'atype': 2},
        'OD2': {'vrad': 1.42, 'atype': 2},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "CYM": {},
    "CYX": {},
    "CYS": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'SG': {'vrad': 1.77, 'atype': 6},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "GLN": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.88, 'atype': 4},
        'CD': {'vrad': 1.61, 'atype': 6},
        'OE1': {'vrad': 1.42, 'atype': 2},
        'NE2': {'vrad': 1.64, 'atype': 3},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "GLU": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.88, 'atype': 4},
        'CD': {'vrad': 1.61, 'atype': 6},
        'OE1': {'vrad': 1.46, 'atype': 2},
        'OE2': {'vrad': 1.42, 'atype': 2},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "GLY": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 6},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "HIE": {},
    "HIP": {},
    "HIS": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.61, 'atype': 5},
        'ND1': {'vrad': 1.64, 'atype': 1},
        'CD2': {'vrad': 1.76, 'atype': 5},
        'CE1': {'vrad': 1.76, 'atype': 5},
        'NE2': {'vrad': 1.64, 'atype': 1},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "ILE": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG1': {'vrad': 1.88, 'atype': 4},
        'CG2': {'vrad': 1.88, 'atype': 4},
        'CD1': {'vrad': 1.88, 'atype': 4},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "LEU": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.88, 'atype': 4},
        'CD1': {'vrad': 1.88, 'atype': 4},
        'CD2': {'vrad': 1.88, 'atype': 4},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "LYS": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.88, 'atype': 4},
        'CD': {'vrad': 1.88, 'atype': 4},
        'CE': {'vrad': 1.88, 'atype': 7},
        'NZ': {'vrad': 1.64, 'atype': 3},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "MET": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.88, 'atype': 4},
        'SD': {'vrad': 1.77, 'atype': 8},
        'CE': {'vrad': 1.88, 'atype': 4},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "PHE": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.88, 'atype': 5},
        'CD1': {'vrad': 1.61, 'atype': 5},
        'CD2': {'vrad': 1.76, 'atype': 5},
        'CE1': {'vrad': 1.76, 'atype': 5},
        'CE2': {'vrad': 1.76, 'atype': 5},
        'CZ': {'vrad': 1.76, 'atype': 5},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "PRO": {
        'N': {'vrad': 1.64, 'atype': 6},
        'CA': {'vrad': 1.88, 'atype': 4},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.88, 'atype': 4},
        'CD': {'vrad': 1.88, 'atype': 4},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "SER": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 6},
        'OG': {'vrad': 1.46, 'atype': 1},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "THR": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 6},
        'OG1': {'vrad': 1.46, 'atype': 1},
        'CG2': {'vrad': 1.88, 'atype': 4},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "TRP": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.61, 'atype': 5},
        'CD1': {'vrad': 1.76, 'atype': 5},
        'CD2': {'vrad': 1.61, 'atype': 5},
        'NE1': {'vrad': 1.64, 'atype': 3},
        'CE2': {'vrad': 1.61, 'atype': 5},
        'CE3': {'vrad': 1.76, 'atype': 5},
        'CZ2': {'vrad': 1.76, 'atype': 5},
        'CZ3': {'vrad': 1.76, 'atype': 5},
        'CH2': {'vrad': 1.76, 'atype': 5},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "TYR": {
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG': {'vrad': 1.61, 'atype': 5},
        'CD1': {'vrad': 1.76, 'atype': 5},
        'CD2': {'vrad': 1.76, 'atype': 5},
        'CE1': {'vrad': 1.76, 'atype': 5},
        'CE2': {'vrad': 1.76, 'atype': 5},
        'CZ': {'vrad': 1.61, 'atype': 5},
        'OH': {'vrad': 1.46, 'atype': 1},
        'default': {'vrad': 0.00, 'atype': 0},
    },
    "VAL": {
        'N  ': {'vrad': 1.64, 'atype': 3},
        'CA ': {'vrad': 1.88, 'atype': 7},
        'C  ': {'vrad': 1.61, 'atype': 6},
        'O  ': {'vrad': 1.42, 'atype': 2},
        'CB ': {'vrad': 1.88, 'atype': 4},
        'CG1': {'vrad': 1.88, 'atype': 4},
        'CG2': {'vrad': 1.88, 'atype': 4},
        'default': {'vrad': 0.00, 'atype': 0},
    }
}


def BONDTYPE(i, j):
    MAXATOMTYPE = 10
    assert i >= 1
    assert i <= MAXATOMTYPE
    assert j >= 1
    assert j <= MAXATOMTYPE

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

def get_nres(molecule):
    """
    get the total number of residues in a molecule, across all chains
    """
    chains = list([molecule.nodes[node]['chain'] for node in molecule.nodes])
    resids = list([molecule.nodes[node]['resid'] for node in molecule.nodes])

    # for all unique chains, get all the resids in that chain
    lens = 0
    for i, c in enumerate(list(set(chains))):
        res = []
        for j, k in enumerate(chains):
            if k == c:
                res.append(j)
        lens += len(list(set([resids[j] for j in res])))
    return lens

def atom_coords(residue_nodes, atomid):
    """
    get the coordinates of an atom indexed internally within a serially numbered residue
    """
    try:
        return residue_nodes[atomid]['position']
    except KeyError:
        return np.array([np.nan, np.nan, np.nan])

def vdw_radius(residue_nodes, atomid):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    res_vdw = PROTEIN_MAP[residue_nodes[atomid]['resname']]
    try:
        atom_vdw = res_vdw[residue_nodes[atomid]['atomname']]['vrad']
    except KeyError:
        atom_vdw = res_vdw['default']['vrad']
    return atom_vdw

def get_residue(molecule, serial_resid):
    """
    get the node entry of a molecule from its serial resid
    """
    residue_nodes = []
    for node in molecule.nodes:
        if molecule.nodes[node]['_res_serial'] == serial_resid:
            residue_nodes.append(molecule.nodes[node])

    return residue_nodes

def make_surface(position, fiba, fibb, vrad):
    """
    Generate points on a sphere using Fibonacci points

    position: centre of sphere
    """
    x, y, z = position[0], position[1], position[2]
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

def model_assertion(res0, res1):
    """
    assert that two residues are from the same model in a pdb

    maybe this can be scrapped given main() is called from run_molecule, so they must be?
    """
    result = []
    for i in res0:
        for j in res1:
            try:
                assert i['mol_idx'] == j['mol_idx']
                result.append(True)
            except AssertionError:
                result.append(False)

    return all(result)

def residue_COM(residue):
    """
    get the centre of mass of a residue

    residue: list of molecule.nodes that make up a residue
    """
    positions = np.zeros((0,3))
    weights = np.zeros(0)
    for i in residue:
        try:
            positions = np.vstack((positions, i['position']))
            weights = np.append(weights, i['mass'])
        except KeyError:
            pass
    return np.average(positions, axis=0, weights=weights)

def get_atype(residue_nodes, atomid):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    res_vdw = PROTEIN_MAP[residue_nodes[atomid]['resname']]
    try:
        atom_vdw = res_vdw[residue_nodes[atomid]['atomname']]['atype']
    except KeyError:
        atom_vdw = res_vdw['default']['atype']
    return atom_vdw


def DISTANCE_C_ALPHA(res0, res1):
    """
    identify the CA atoms within a residue and calculate the distance between them
    """
    atom0 = None
    atom1 = None
    for i in res0:
        if i['atomname'] == 'CA':
            atom0 = i
            continue
    for i in res1:
        if i['atomname'] == 'CA':
            atom1 = i
            continue
    if (atom0 is not None) and (atom1 is not None):
        return euclidean(atom0['position'], atom1['position'])*10 #this distance is working, because the output distances where the contacts are found are matching
    else:
        return None

def atom2res(nresidues, atom_array, mask, onemax=False):
    '''
    count the atom entries in each residue and return an array at the residue level
    '''
    out = np.zeros((nresidues, nresidues), dtype=int)
    # map all atom back to residues
    for i in range(nresidues):
        find0 = np.where(mask == i)[0]
        for j in range(nresidues):
            find1 = np.where(mask == j)[0]
            out[i, j] = np.count_nonzero(atom_array[find0][:, find1])

    if onemax:
        out[np.where(out > 0)] = 1
    return out

def res2atom(nresidues, res_array, mask):
    '''
    take an array with residue level data and repeat the entries over each atom within the residue
    '''
    # find out what the values are, and how many of them there are
    unique_values, counts = np.unique(mask, return_counts=True)

    assert len(unique_values) == nresidues

    out = np.zeros((len(mask), len(mask)))
    start0 = 0
    for i, j in zip(unique_values, counts):
        start1 = 0
        for k, l in zip(unique_values, counts):
            target_value = res_array[i, k]
            out[start0:start0 + j, start1:start1 + l] = target_value
            start1 += l
        start0 += j

    return out

def main(molecule):
    start0 = time.time()
    # get the number of residues
    nresidues = get_nres(molecule)

    # Fibonacci sequence generation. Can make arguments for this later, but 14 is what's usually needed
    fib = 14
    fiba, fibb = 0, 1
    for _ in range(fib):
        fiba, fibb = fibb, fiba + fibb

    alpha = 1.24  # Enlargement factor for attraction effects
    water_radius = 2.80  # Radius of a water molecule in A

    # loop over all atoms in the protein to get atom information
    coords = []
    vdw_list = []
    mask = []
    atypes = []
    coms = []
    for i in range(nresidues):
        res0 = get_residue(molecule, i)
        mask.append([str(i)]*len(res0))
        coms.append(residue_COM(res0))
        for j in range(len(res0)):
            coords.append(atom_coords(res0, j))
            vdw_list.append(vdw_radius(res0, j))
            atypes.append(get_atype(res0, j))
    mask = np.array([x for xs in mask for x in xs], dtype=int)

    # definitely need the *10 here to make sure these distances are correct.
    com_distances = res2atom(nresidues, cdist(np.array(coms), np.array(coms))*10, mask)

    setup_time = time.time()

    # calculate all interatomic distances in the protein
    distances = cdist(np.array(coords), np.array(coords))*10
    distance_time = time.time()

    # generate fibonacci spheres for all atoms. This is a big time limiting step atm
    spheres = np.stack([make_surface(i, fiba, fibb, water_radius + j) for i, j in zip(coords, vdw_list)])
    sphere_time = time.time()
    # print(f'sphere time {sphere_time - distance_time}')

    # # vdw is nxn array of all pairs of atomic vdws
    vdw_sum_array = np.array(vdw_list)[:, np.newaxis] + np.array(vdw_list)[np.newaxis, :]

    # Enlarged overlap (OV) contact
    overlapcounter = np.zeros(distances.shape, dtype=int)
    overlapcounter[distances < (vdw_sum_array * alpha)] = 1  # this still needs correcting for intra-residue contacts
    overlapcounter[np.where(com_distances < 3.5 * 4)] = 0

    contactcounter = np.zeros(distances.shape, dtype=int)
    stabilizercounter = np.zeros(distances.shape, dtype=int)
    destabilizercounter = np.zeros(distances.shape, dtype=int)

    # this loop takes about as long as the sphere point generation, pretty sure it could be sped up somehow
    # find where csu contacts are possibly relevant
    csu_criteria = np.zeros(distances.shape)
    csu_targets = np.where(distances < (vdw_sum_array * water_radius))
    for i in np.unique(csu_targets[0]):
        group = np.stack(csu_targets).T[csu_targets[0] == i]
        target_coords = []
        for j in group:
            if (not mask[j[0]] == mask[j[1]]) and (com_distances[j[0], j[1]] < 3.5 * 4):
                target_coords.append(coords[j[1]])
        if len(target_coords) > 0:
            contact_distances = cdist(spheres[group.T[0]][0], np.stack(target_coords))
            # this now gives the correct indexing
            csu_criteria[i, np.unravel_index(contact_distances.argmin(), contact_distances.shape)[1]] = 1
    print(csu_criteria)
    # csu_time = time.time()
    # # print(f'csu time {csu_time - sphere_time}')
    #
    # for i in range(distances.shape[0]):
    #     for j in range(distances.shape[1]):
    #         at1, at2 = atypes[i], atypes[j]
    #         if at1 > 0 and at2 > 0:
    #             contactcounter[i, j] += 1
    #             btype = BONDTYPE(at1, at2)
    #             if btype <= 4:
    #                 stabilizercounter[i, j] += 1
    #             if btype == 5:
    #                 destabilizercounter[i, j] += 1
    # counting = time.time()
    # # print(f'counting time {counting - csu_time}')
    #
    # overlap_residues = atom2res(nresidues, overlapcounter, mask, onemax=True)
    # contact_residues = atom2res(nresidues, contactcounter, mask)
    # stabiliser_residues = atom2res(nresidues, stabilizercounter, mask)
    # destabiliser_residues = atom2res(nresidues, destabilizercounter, mask)
    #
    # finish = time.time()
    # # print(f'finish {finish - counting}')
    #
    # with open('test.out', 'w') as f:
    #     count = 0
    #     for i1 in range(nresidues):
    #         for i2 in range(nresidues):
    #             res0 = get_residue(molecule, i1)
    #             res1 = get_residue(molecule, i2)
    #             if not (i1 == i2) and model_assertion(res0, res1):
    #                 over = overlap_residues[i1, i2]
    #                 cont = contact_residues[i1, i2]
    #                 stab = stabiliser_residues[i1, i2]
    #                 dest = destabiliser_residues[i1, i2]
    #                 ocsu = stab
    #                 rcsu = stab - dest
    #
    #                 if over > 0 or cont > 0:
    #                     count += 1
    #                     msg = (f"R {count:6d} "
    #                            f"{i1 + 1:5d} {res0[0]['resname']:4} {res0[0]['chain']:1} {res0[0]['resid']:4d}    "
    #                            f"{i2 + 1:5d} {res1[0]['resname']:4} {res1[0]['chain']:1} {res1[0]['resid']:4d}    "
    #                            f"{DISTANCE_C_ALPHA(res0, res1):8.4f}     "
    #                            f"{over} {1 if cont != 0 else 0} {1 if ocsu != 0 else 0} {1 if rcsu > 0 else 0} "
    #                            f"{rcsu:6d}  {cont:6d} {res0[0]['mol_idx']:4d}\n")
    #                     f.writelines(msg)
    #













    # at1 = get_atype(res0, j1)
    # at2 = get_atype(get_residue(molecule, int(s[3])), int(s[4]))
    # if at1 > 0 and at2 > 0:
    #     contactcounter[i1, int(s[3])] += 1
    #     btype = BONDTYPE(at1, at2)
    #     if btype <= 4:
    #         stabilizercounter[i1, int(s[3])] += 1
    #     if btype == 5:
    #         destabilizercounter[i1, int(s[3])] += 1


            # base_atom_vdw = vdw_radius(res0, j1)
            # if base_atom_coords is not None:
                # generate the Fibonacci surface points on a sphere
                # surface = make_surface(base_atom_coords, fiba, fibb, base_atom_vdw + water_radius)
                # loop over all the residues a second time
                # for i2 in range(nresidues):
                #     res1 = get_residue(molecule, i2)
                #     if model_assertion(res0, res1) and close_residues(res0, res1):
                #         for j2 in range(len(res1)):
                            # if not (i1 == i2 and j1 == j2):
                            #     target_atom_coords = atom_coords(res1, j2)
                            #     target_atom_vdw = vdw_radius(res1, j2)
    #                             if target_atom_coords is not None:
    #                                 distance = euclidean(base_atom_coords, target_atom_coords)*10
    #
    #                                 # Enlarged overlap (OV) contact
    #                                 if distance <= ((base_atom_vdw + target_atom_vdw) * alpha):
    #                                     overlapcounter[i1, i2] = 1
    #
    #                                 # CSU contacts
    #                                 if distance <= ((base_atom_vdw + target_atom_vdw) * water_radius):
    #                                     for k in range(fibb):
    #                                         s = surface[k]
    #                                         if (euclidean(s[:3], target_atom_coords) < target_atom_vdw + water_radius) \
    #                                                 and distance <= s[5]:
    #                                             s[3] = i2
    #                                             s[4] = j2
    #                                             s[5] = distance
    #
    #             for k in range(fibb):
    #                 s = surface[k]
    #                 if s[3] >= 0 and s[4] >= 0:
    #                     at1 = get_atype(res0, j1)
    #                     at2 = get_atype(get_residue(molecule, int(s[3])), int(s[4]))
    #                     if at1 > 0 and at2 > 0:
    #                         contactcounter[i1, int(s[3])] += 1
    #                         btype = BONDTYPE(at1, at2)
    #                         if btype <= 4:
    #                             stabilizercounter[i1, int(s[3])] += 1
    #                         if btype == 5:
    #                             destabilizercounter[i1, int(s[3])] += 1
    #
    # with open('test.out', 'w') as f:
    #     count = 0
    #     for i1 in range(nresidues):
    #         for i2 in range(nresidues):
    #             res0 = get_residue(molecule, i1)
    #             res1 = get_residue(molecule, i2)
    #             if not (i1 == i2) and model_assertion(res0, res1):
    #                 over = overlapcounter[i1, i2]
    #                 cont = contactcounter[i1, i2]
    #                 stab = stabilizercounter[i1, i2]
    #                 dest = destabilizercounter[i1, i2]
    #                 ocsu = stab
    #                 rcsu = stab - dest
    #
    #                 if over > 0 or cont > 0:
    #                     count += 1
    #                     msg = (f"R {count:6d} "
    #                            f"{i1 + 1:5d} {res0[0]['resname']:4} {res0[0]['chain']:1} {res0[0]['resid']:4d}    "
    #                            f"{i2 + 1:5d} {res1[0]['resname']:4} {res1[0]['chain']:1} {res1[0]['resid']:4d}    "
    #                            f"{DISTANCE_C_ALPHA(res0, res1):8.4f}     "
    #                            f"{over} {1 if cont != 0 else 0} {1 if ocsu != 0 else 0} {1 if rcsu > 0 else 0} "
    #                            f"{rcsu:6d}  {cont:6d} {res0[0]['mol_idx']:4d}\n")
    #                     f.writelines(msg)

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

    def run_molecule(self, molecule):
        self.system.go_params["go_map"].append(read_go_map(file_path=self.path))
        main(molecule)
        return molecule

    def run_system(self, system):
        """
        Process `system`.

        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        self.system = system
        super().run_system(system)
