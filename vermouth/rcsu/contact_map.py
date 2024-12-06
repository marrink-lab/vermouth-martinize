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
        'CA': {'vrad': 1.88, 'atype': 7},
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
        'CD': {'vrad': 1.88, 'atype': 4},
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
        'N': {'vrad': 1.64, 'atype': 3},
        'CA': {'vrad': 1.88, 'atype': 7},
        'C': {'vrad': 1.61, 'atype': 6},
        'O': {'vrad': 1.42, 'atype': 2},
        'CB': {'vrad': 1.88, 'atype': 4},
        'CG1': {'vrad': 1.88, 'atype': 4},
        'CG2': {'vrad': 1.88, 'atype': 4},
        'default': {'vrad': 0.00, 'atype': 0},
    }
}


def get_vdw_radius(resname, atomname):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    res_vdw = PROTEIN_MAP[resname]
    try:
        atom_vdw = res_vdw[atomname]['vrad']
    except KeyError:
        atom_vdw = res_vdw['default']['vrad']
    return atom_vdw


def get_atype(resname, atomname):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    res_vdw = PROTEIN_MAP[resname]
    try:
        atom_vdw = res_vdw[atomname]['atype']
    except KeyError:
        atom_vdw = res_vdw['default']['atype']
    return atom_vdw


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


def res2atom(arrin, residues, nresidues):
    '''
    take an array with residue level data and repeat the entries over
    each atom within the residue

    would be nice to do this with list comprension but Ican't work out how
    to do something like:

    [np.tile(res_dists[i-1,j-1],
             np.where(residues == i)[0].size)
        for i in range(1, nresidues+1)
        for j in range(1, nresidues+1)
    ]

    to get it correct in the 2 dimensional way we actually need.

    At the moment we only need this function once
    (to get the residue COGs foreach atom)
    so it's not too much of a limiting factor, but something to optimise
    better in future

    '''
    # find out how many residues we have, and how many atoms are in each of them
    unique_values, counts = np.unique(residues, return_counts=True)

    assert len(unique_values) == nresidues

    out = np.zeros((len(residues), len(residues)))
    start0 = 0
    for i, j in zip(unique_values - 1, counts):
        start1 = 0
        for k, l in zip(unique_values - 1, counts):
            target_value = arrin[i, k]
            out[start0:start0 + j, start1:start1 + l] = target_value.sum()
            start1 += l
        start0 += j

    return out


def atom2res(arrin, residues, nresidues, norm=False):
    '''
    take an array with atom level data and sum the entries over within the residue
    '''
    out = np.array([int(arrin[np.where(residues == i)[0], np.where(residues == j)[0][:, np.newaxis]].sum())
                    for i in range(1, nresidues + 1)
                    for j in range(1, nresidues + 1)]).reshape((nresidues, nresidues))
    if norm:
        out[out > 0] = 1

    return out


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

def calculate_contact_map(mol):

    # some initial definitions of variables that we need
    fib = 14
    fiba, fibb = 0, 1
    for _ in range(fib):
        fiba, fibb = fibb, fiba + fibb

    alpha = 1.24  # Enlargement factor for attraction effects
    water_radius = 2.80  # Radius of a water molecule in A

    # 1) Set up the basic information that we need to calculate the contact map
    residues = []
    coords = []
    vdw_list = []
    atypes = []
    resnames = []
    ca_pos = []
    chains = []
    for node in mol.nodes:
        try:
            coords.append(mol.nodes[node]['position'] * 10)  #  need *10 here because of the way Vermouth read coords
            residues.append(mol.nodes[node]['resid'])
            resnames.append(mol.nodes[node]['resname'])
            vdw_list.append(get_vdw_radius(mol.nodes[node]['resname'],
                                           mol.nodes[node]['atomname']))
            atypes.append(get_atype(mol.nodes[node]['resname'],
                                    mol.nodes[node]['atomname']))
            if mol.nodes[node]['atomname'] == "CA":
                ca_pos.append(mol.nodes[node]['position'])
            chains.append(mol.nodes[node]['chain'])

        except KeyError:
            pass

    residues = np.array(residues)
    vdw_list = np.array(vdw_list)
    atypes = np.array(atypes)

    # 2) find the number of residues that we have
    nresidues = len(list(set(residues)))

    # 4) set up final bits of information
    # sums of vdw pairs for each atom we have
    vdw_sum_array = vdw_list[:, np.newaxis] + vdw_list[np.newaxis, :]

    # distances between all the atoms that we have
    atomic_distances = cdist(np.stack(coords), np.stack(coords))

    # array with 1 on the diagonal, so we can exclude the self atoms
    diagonal_ones = np.diagflat(np.ones(atomic_distances.shape[0], dtype=int))

    # get the coordinates of the centres of geometry for each residue
    res_cogs = np.stack(
        [np.stack(coords[np.where(residues == i)[0][0]:np.where(residues == i)[0][-1]]).mean(axis=0) for i in
         range(1, nresidues + 1)])
    res_dists = res2atom(cdist(res_cogs, res_cogs), residues, nresidues)

    # 5) find atoms which meet the overlap criterion
    over = np.zeros_like(atomic_distances)
    overlaps = np.where((atomic_distances <= (vdw_sum_array * alpha)) & (diagonal_ones != 1) & (res_dists < 14))
    over[overlaps[0], overlaps[1]] = 1

    # 6) set up the surface overlap criterion
    # generate fibonacci spheres for all atoms.
    # can't decide whether quicker/better for memory to generate all in one go here
    # or incorporate into the loop. left
    spheres = np.stack([make_surface(i, fiba, fibb, water_radius + j) for i, j in zip(coords, vdw_list)])
    surface_overlaps = np.where(
        (atomic_distances <= (vdw_sum_array + water_radius)) & (diagonal_ones != 1) & (res_dists < 14))
    # find which atoms are uniquely involved as base points
    base_points = np.unique(surface_overlaps[0])

    hit_results = np.ones((spheres.shape[0], spheres.shape[1]), dtype=int) * -1

    # loop over all base points
    for base_point in base_points:

        # generate the base point sphere now if we didn't earlier.
        # sphere = make_surface(coords[base_point], fiba, fibb, vdw_list[base_point] + water_radius)
        # get the target points
        target_points = surface_overlaps[1][np.where(surface_overlaps[0] == base_point)[0]]
        # array of all the target point coordinates
        target_point_coords = np.stack(coords)[target_points]
        # distances between the points on the base sphere surface and the target point coordinates
        surface_to_point = cdist(spheres[base_point], target_point_coords)
        # surface_to_point = cdist(sphere, target_point_coords)
        # cutoff distances for each of the target points
        target_distances = vdw_list[target_points] + water_radius

        for i, j in enumerate(surface_to_point):
            '''
            first find where the radius condition is met, i.e. where the distance between
            the target point and this point on the surface is smaller than the vdw radius
            of the target point
            '''
            radius_condition = j < target_distances
            if any(radius_condition):
                '''
                For all the points that meet this condition, look at the distance between the 
                target point and the base point
                '''
                distances_to_compare = atomic_distances[base_point][target_points[radius_condition]]
                '''
                the point that we need is the point with the smallest distance
                '''
                point_needed = target_points[radius_condition][distances_to_compare.argmin()]
                hit_results[base_point, i] = point_needed

    contactcounter_1 = np.zeros_like(atomic_distances)
    stabilisercounter_1 = np.zeros_like(atomic_distances)
    destabilisercounter_1 = np.zeros_like(atomic_distances)

    for i, j in enumerate(hit_results):
        for k in j:
            if k >= 0:
                at1 = atypes[i]
                at2 = atypes[k]
                if (at1 > 0) and (at2 > 0):
                    contactcounter_1[i, k] += 1
                    btype = BONDTYPE(at1, at2)
                    if btype <= 4:
                        stabilisercounter_1[i, k] += 1
                    if btype == 5:
                        destabilisercounter_1[i, k] += 1

    overlapcounter_2 = atom2res(over, residues, nresidues, norm=True)
    contactcounter_2 = atom2res(contactcounter_1, residues, nresidues)
    stabilisercounter_2 = atom2res(stabilisercounter_1, residues, nresidues)
    destabilisercounter_2 = atom2res(destabilisercounter_1, residues, nresidues)

    resnames = np.array(resnames)[np.unique(residues, return_index=True)[1]]
    resids = np.array(residues)[np.unique(residues, return_index=True)[1]]

    # this to write out the file if needed
    # with open('contact_map_out.out', 'w') as f:
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
    #                 count += 1
    #                 msg = (f"R {int(count):6d} "
    #                        f"{int(i1 + 1):5d}  {resnames[i1]:3s} A {int(resids[i1]):4d}    "
    #                        f"{int(i2 + 1):5d}  {resnames[i2]:3s} A {int(resids[i2]):4d}    "
    #                        f"{euclidean(ca_pos[i1], ca_pos[i2]) * 10:9.4f}     "
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
                if over == 1 or (over == 0 and rcsu == 1):
                    # this is a OV or rCSU contact we take it
                    chain_i1 = chains[np.where(residues == i1+1)[0][0]]
                    chain_i2 = chains[np.where(residues == i2+1)[0][0]]
                    contacts.append((i1+1, chain_i1, i2+1, chain_i2))
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

    def run_molecule(self, molecule):
        if self.path is None:
            self.system.go_params["go_map"].append(calculate_contact_map(molecule))
        else:
            self.system.go_params["go_map"].append(read_go_map(file_path=self.path))
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
