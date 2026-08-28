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
import scipy.sparse as sp
from scipy.spatial.distance import euclidean
from scipy.spatial import cKDTree as KDTree
from ..graph_utils import make_residue_graph
from vermouth.file_writer import deferred_open
from ..log_helpers import StyleAdapter, get_logger
from collections import defaultdict
from vermouth import __version__ as VERSION
from pathlib import Path

LOGGER = StyleAdapter(get_logger(__name__))

# BOND TYPE
# Types of contacts:
# HB -- 1 -- hydrogen-bond
# PH -- 2 -- hydrophobic
# AR -- 3 -- aromatic - contacts between aromatic rings
# IB -- 4 -- ionic bridge - contacts created by two atoms with different charges
# DC -- 5 -- destabilizing contact - contacts which are in general repulsive
# OT -- 6 -- denotes negligible other contacts.
# 1-HB,2-PH,3-AR,4-IP,5-DC,6-OT
BOND_TYPE = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 5, 5, 6, 6, 6, 1, 1],
                      [0, 1, 5, 1, 5, 5, 6, 6, 6, 1, 5],
                      [0, 1, 1, 5, 5, 5, 6, 6, 6, 5, 1],
                      [0, 5, 5, 5, 2, 2, 6, 6, 6, 5, 5],
                      [0, 5, 5, 5, 2, 3, 6, 6, 6, 5, 5],
                      [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [0, 1, 1, 5, 5, 5, 6, 6, 6, 5, 4],
                      [0, 1, 5, 1, 5, 5, 6, 6, 6, 4, 5]])

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


def _get_vdw_radius(resname, atomname):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    try:
        res_vdw = PROTEIN_MAP[resname]
    except KeyError:
        return 0.00

    try:
        atom_vdw = res_vdw[atomname]
    except KeyError:
        atom_vdw = res_vdw['default']
    return atom_vdw['vrad']


def _lookup_atom_type(resname, atomname):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    try:
        res_vdw = PROTEIN_MAP[resname]
    except KeyError:
        return 0

    try:
        atom_vdw = res_vdw[atomname]
    except KeyError:
        atom_vdw = res_vdw['default']

    return atom_vdw['atype']


def _make_fibonacci_sphere(position, fiba, fibb, vrad):
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

    k = np.arange(fibb)
    phi_aux = (np.arange(1, fibb+1) * fiba) % fibb
    phi_aux[phi_aux == 0] = fibb
    theta = np.arccos(1.0 - 2.0 * k / fibb)
    phi = 2.0 * np.pi * phi_aux / fibb
    surface_x = x + vrad * np.sin(theta) * np.cos(phi)
    surface_y = y + vrad * np.sin(theta) * np.sin(phi)
    surface_z = z + vrad * np.cos(theta)
    surface = np.stack((surface_x, surface_y, surface_z), axis=-1)

    return surface


def _aggregate_atoms_to_residues(arrin, nresidues, atom_map, norm=False):
    '''
    take an array with atom level data and sum the entries over within the residue
    '''
    if not sp.issparse(arrin):
        arrin = sp.csr_matrix(arrin)
    natoms = arrin.shape[0]
    rows, cols = [], []
    for res_i, atom_idxs in atom_map.items():
        for atom_idx in atom_idxs:
            rows.append(res_i)
            cols.append(int(atom_idx))
    P = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nresidues, natoms))
    out = (P @ arrin @ P.T).tocsr()
    if norm:
        out.eliminate_zeros()
        out.data[:] = 1.0
    return out


def _extract_contact_inputs(molecule):
    """
    get the atom attributes that we need to calculate the contacts
    """

    G = make_residue_graph(molecule)

    resids = []
    chains = []
    resnames = []
    positions_all = []
    ca_pos = []
    vdw_list = []
    atypes = []
    res_serial = []
    res_idx = []
    for residue in G.nodes:
        # we only need these for writing at the end
        resnames.append(G.nodes[residue]['resname'])
        resids.append(G.nodes[residue]['resid'])
        chains.append(G.nodes[residue]['chain'])
        res_idx.append(G.nodes[residue]['_res_serial'])
        subgraph = G.nodes[residue]['graph']

        for atom in sorted(subgraph.nodes):
            position = subgraph.nodes[atom].get('position', [np.nan]*3)
            if np.isfinite(position).all():
                res_serial.append(subgraph.nodes[atom]['_res_serial'])

                positions_all.append(subgraph.nodes[atom]['position'] * 10)

                vdw_list.append(_get_vdw_radius(subgraph.nodes[atom]['resname'],
                                                subgraph.nodes[atom]['atomname']))
                atypes.append(_lookup_atom_type(subgraph.nodes[atom]['resname'],
                                         subgraph.nodes[atom]['atomname']))

                if subgraph.nodes[atom]['atomname'] == 'CA':
                    ca_pos.append(subgraph.nodes[atom]['position'])


    vdw_list = np.array(vdw_list)
    atypes = np.array(atypes)
    coords = np.stack(positions_all)
    res_serial = np.array(res_serial)

    resids = np.array(resids)
    chains = np.array(chains)
    resnames = np.array(resnames)
    res_idx = np.array(res_idx)

    # 2) find the number of residues that we have
    nresidues = len(G)

    LOGGER.debug("Extracted {} atoms from {} residues", len(positions_all), nresidues)
    return vdw_list, atypes, coords, res_serial, resids, chains, resnames, res_idx, ca_pos, nresidues, G

def _calculate_ov_contacts(coo, vdw_list, natoms, vdw_max, alpha=1.24):
    """
    Find enlarged (OV) overlap contacts

    coo: scipy.sparse.coo_matrix
        precomputed sparse pairwise-distance matrix (COO format) of the
        input coordinates, out to at least a radius of 2 * vdw_max * alpha.
        This is shared with _calculate_csu_contacts so that the expensive
        KDTree-based distance query is only performed once per molecule
        instead of once per contact type.
    vdw_list: list
        list of vdw radii of the input coordinates
    natoms: int
        number of atoms in the molecule
    vdw_max: float
        maximum possible vdw radius of atoms
    alpha: float
        Enlargement factor for attraction effects
    """
    vdw_list = np.asarray(vdw_list)
    vdw_sum = alpha * (vdw_list[coo.row] + vdw_list[coo.col])
    # cutoff_ov is OV's own cutoff radius (2 * vdw_max * alpha). In normal
    # usage every per-atom vdw radius is <= vdw_max, so vdw_sum can never
    # exceed cutoff_ov and this comparison is redundant with the one
    # above. It is kept explicit so that behaviour is independent of
    # whichever (possibly larger) radius was used to build the shared
    # `coo` distance matrix passed in from _compute_residue_contacts.
    cutoff_ov = 2 * vdw_max * alpha
    keep = (coo.row < coo.col) & (coo.data < vdw_sum) & (coo.data < cutoff_ov)
    rows = coo.row[keep]
    cols = coo.col[keep]
    LOGGER.debug("Found {} OV overlapping atom pairs", len(rows))
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    return sp.csr_matrix(
        (np.ones(len(all_rows), dtype=np.float32), (all_rows, all_cols)),
        shape=(natoms, natoms)
    )

def _calculate_csu_contacts(coords, vdw_list, fiba, fibb, natoms, coo, vdw_max, water_radius=2.80):
    """
    Calculate contacts of structural units (CSU)

    coords: Nx3 numpy array
        coordinates of atoms in the molecule
    vdw_list: list
        vdw radii of the atoms in the molecule
    fiba, fibb: int
        n-1th and nth fibonacci numbers from which to generate points on a sphere around the input coordinate
    natoms: int
        number of atoms in the molecule
    coo: scipy.sparse.coo_matrix
        precomputed sparse pairwise-distance matrix (COO format) of the
        input coordinates, out to at least a radius of 2 * vdw_max +
        water_radius. Shared with _calculate_ov_contacts; see that
        function's docstring.
    vdw_max: float
        maximum possible vdw radius of atoms
    water_radius: float
        radius of water molecule in A

    Returns:
    hit_results: natoms x fibb np.array
        each i,j entry is the index of the atom in coords which is the closest atom to atom i at index j of the
        fibonacci sphere

    """
    vdw_arr = np.asarray(vdw_list)
    LOGGER.debug("Computing CSU contacts for {} atoms using {} Fibonacci sphere points per atom",
                 natoms, fibb)
    hit_results = np.full((natoms, fibb), -1, dtype=np.int32)

    # Vectorised pre-filter: remove self-pairs and pairs beyond the per-atom cutoff
    vdw_sum = vdw_arr[coo.row] + vdw_arr[coo.col] + water_radius
    valid = (coo.row != coo.col) & (coo.data < vdw_sum)
    valid_rows = coo.row[valid]
    valid_cols = coo.col[valid]
    valid_dists = coo.data[valid]

    if len(valid_rows) == 0:
        return hit_results

    # Sort by (idx, jdx) to group neighbours per base atom, preserving original iteration order
    sort_order = np.lexsort((valid_cols, valid_rows))
    sorted_rows = valid_rows[sort_order]
    sorted_cols = valid_cols[sort_order]
    sorted_dists = valid_dists[sort_order]

    unique_idx, first_occ = np.unique(sorted_rows, return_index=True)
    ends = np.append(first_occ[1:], len(sorted_rows))

    # The Fibonacci sphere sample *directions* are identical for every
    # atom - only each atom's centre (its position) and radius (its vdw
    # radius + water_radius) differ, and those enter only as a
    # translation and a uniform scaling. Since translation and uniform
    # scaling preserve nearest-neighbour relationships (up to the scale
    # factor on the radius), we can build a single KDTree once on the
    # *unit* sphere directions and, for each atom, transform its
    # candidate neighbours into that atom's local (centred, unit-radius)
    # frame before querying the shared tree. Previously a fresh KDTree of
    # `fibb` points was built from scratch for every one of the natoms
    # atoms; the per-tree construction overhead, repeated natoms times,
    # was the dominant cost of this function for large systems.
    unit_sphere = _make_fibonacci_sphere(np.zeros(3), fiba, fibb, 1.0)
    unit_tree = KDTree(unit_sphere)

    for idx, start, end in zip(unique_idx, first_occ, ends):
        neighbors = sorted_cols[start:end]
        dists = sorted_dists[start:end]

        radius_i = vdw_arr[idx] + water_radius
        local_points = (coords[neighbors] - coords[idx]) / radius_i
        local_radii = (vdw_arr[neighbors] + water_radius) / radius_i

        dists_counter = np.full(fibb, np.inf)

        # Query all neighbours in one batched call with per-neighbour radii
        all_res = unit_tree.query_ball_point(local_points, local_radii)

        for jdx, dist, res in zip(neighbors, dists, all_res):
            res = np.asarray(res, dtype=np.intp)
            if len(res) > 0:
                to_fill = res[dist < dists_counter[res]]
                if len(to_fill) > 0:
                    dists_counter[to_fill] = dist
                    hit_results[idx, to_fill] = jdx

    return hit_results


def _classify_contact_types(hit_results, natoms, atypes):
    """
    From CSU contacts, establish contact types from atomtypes

    This is a vectorised replacement for a previous implementation that
    accumulated results in Python dictionaries keyed by (i, k) tuples.
    For large systems (many atoms x many Fibonacci sphere sample points
    per atom), that dict-based approach could hold millions of entries,
    each carrying substantial Python object overhead well beyond the
    actual data being stored, which was the dominant source of memory
    (and time) consumption when computing Go contacts on large structures.

    Here, the hit matrix is flattened once and filtered with array
    operations, then the (row, col) count matrices are built directly as
    sparse matrices: converting a COO matrix with repeated (row, col)
    coordinates to CSR sums the duplicate entries automatically, which is
    exactly the "count of occurrences" semantics the original dict-based
    accumulation implemented.

    hit_results: NxM ndarray
        array for N atoms in molecule for M fibonnaci points on each atom.
        Each i,j entry is the index of the atom which is the closest contact to i
    natoms: int
        number of atoms in the molecule
    atypes: array
        list of the atomtypes of each atom in the molecule
    """
    fibb = hit_results.shape[1]

    # Deduplicate (i, k) pairs and count occurrences per atom row, rather
    # than flattening the whole natoms x fibb hit matrix up front. In
    # practice the same (i, k) pair recurs many times within a row (one
    # large neighbouring atom k covers several of atom i's Fibonacci
    # sample directions), so the number of unique pairs per atom is
    # normally a small fraction of fibb. Working row-by-row keeps peak
    # memory proportional to the number of unique contacts (matching the
    # dict-based implementation this replaces, `dict.get(key, 0) + 1`,
    # which deduplicated incrementally) instead of to the much larger
    # number of raw hits, while still doing the per-row work with
    # vectorised NumPy calls (np.unique on at most `fibb` elements) rather
    # than a Python-level loop over every individual hit.
    rows_chunks = []
    cols_chunks = []
    counts_chunks = []
    for i in range(natoms):
        if atypes[i] <= 0:
            continue
        row = hit_results[i]
        row = row[row >= 0]
        if row.size == 0:
            continue
        row = row[atypes[row] > 0]
        if row.size == 0:
            continue
        uniq, counts = np.unique(row, return_counts=True)
        rows_chunks.append(np.full(uniq.shape, i, dtype=np.int32))
        cols_chunks.append(uniq.astype(np.int32))
        counts_chunks.append(counts.astype(np.int32))

    def _to_csr(rows, cols, data):
        if len(rows) == 0:
            return sp.csr_matrix((natoms, natoms), dtype=np.int32)
        return sp.csr_matrix((data, (rows, cols)), shape=(natoms, natoms), dtype=np.int32)

    if not rows_chunks:
        LOGGER.debug("Classified 0 atom contact pairs (0 stabilising, 0 destabilising)")
        empty = sp.csr_matrix((natoms, natoms), dtype=np.int32)
        return empty, empty.copy(), empty.copy()

    u_rows = np.concatenate(rows_chunks)
    u_cols = np.concatenate(cols_chunks)
    counts = np.concatenate(counts_chunks)

    contact_csr = _to_csr(u_rows, u_cols, counts)

    # Bond type depends only on the (fixed) atom types of the pair, so it
    # is identical for every occurrence of a given (i, k) pair: compute it
    # once per unique pair rather than once per raw occurrence.
    btypes = BOND_TYPE[atypes[u_rows], atypes[u_cols]]
    stab_mask = btypes <= 4
    destab_mask = btypes == 5

    stab_csr = _to_csr(u_rows[stab_mask], u_cols[stab_mask], counts[stab_mask])
    destab_csr = _to_csr(u_rows[destab_mask], u_cols[destab_mask], counts[destab_mask])

    LOGGER.debug("Classified {} atom contact pairs ({} stabilising, {} destabilising)",
                 contact_csr.nnz, stab_csr.nnz, destab_csr.nnz)

    return contact_csr, stab_csr, destab_csr

def _build_residue_atom_index(res_serial):

    atom_map = defaultdict(list)
    for atom_idx, res_idx in enumerate(res_serial):
        atom_map[res_idx].append(atom_idx)
    for key, value in atom_map.items():
        atom_map[key] = np.array(value)

    return atom_map

def _compute_residue_contacts(vdw_list, atypes, coords, res_serial, nresidues):
    """
    run the contact calculation functions

    vdw_list: np.array
        list of the vdw radii of the atoms in the system
    atypes: np.array
        list of the atom types in the system to determine the nature of contacts
    coords: nx3 array
        coordinates of all the atoms in the system
    res_serial: np.array
        list of the serial residue number of each atom in the system
    nresidues: int
        number of residues in the system
    """

    # some initial definitions of variables that we need
    fib = 14
    fiba, fibb = 0, 1
    for _ in range(fib):
        fiba, fibb = fibb, fiba + fibb

    natoms = len(coords)

    vdw_max = max(item['vrad'] for atoms in PROTEIN_MAP.values() for item in atoms.values())

    coords_tree = KDTree(coords)

    # OV and CSU contacts both start from a pairwise distance query on the
    # same coordinate set; the only difference is the cutoff radius used
    # to prune candidate pairs. Computing this once (at the larger of the
    # two cutoffs) and reusing it avoids running the expensive KDTree
    # distance query twice, which previously roughly doubled both the
    # runtime and peak memory of this step for large systems.
    alpha = 1.24
    water_radius = 2.80
    cutoff_ov = 2 * vdw_max * alpha
    cutoff_csu = (2 * vdw_max) + water_radius
    max_cutoff = max(cutoff_ov, cutoff_csu)

    LOGGER.debug("Computing shared pairwise distance matrix for {} atoms (cutoff={:.2f} A)",
                 natoms, max_cutoff)
    sparse_dm = coords_tree.sparse_distance_matrix(coords_tree, max_cutoff)
    coo = sparse_dm.tocoo()

    LOGGER.debug("Computing OV overlap contacts for {} atoms", natoms)
    over = _calculate_ov_contacts(coo, vdw_list, natoms, vdw_max, alpha=alpha)

    LOGGER.debug("Computing CSU surface contacts")
    hit_results = _calculate_csu_contacts(coords, vdw_list, fiba, fibb, natoms, coo, vdw_max, water_radius=water_radius)

    LOGGER.debug("Classifying contact types by bond chemistry")
    contactcounter_1, stabilisercounter_1, destabilisercounter_1 = _classify_contact_types(hit_results, natoms, atypes)

    atom_map = _build_residue_atom_index(res_serial)

    LOGGER.debug("Projecting atom contacts to residue level ({} atoms -> {} residues)", natoms, nresidues)
    overlapcounter_2 = _aggregate_atoms_to_residues(over, nresidues, atom_map, norm=True)
    contactcounter_2 = _aggregate_atoms_to_residues(contactcounter_1, nresidues, atom_map)
    stabilisercounter_2 = _aggregate_atoms_to_residues(stabilisercounter_1, nresidues, atom_map)
    destabilisercounter_2 = _aggregate_atoms_to_residues(destabilisercounter_1, nresidues, atom_map)

    return overlapcounter_2, contactcounter_2, stabilisercounter_2, destabilisercounter_2


def _filter_rcsu_contacts(overlaps, contacts, stabilisers, destabilisers, res_idx, G):
    '''
    Generate contacts list from the contact arrays calculated

    nresidues: int
        number of residues in the molecule
    overlaps: sparse or dense array
        nresidues x nresidues array of OV contacts in the molecule
    contacts: sparse or dense array
        nresidues x nresidues array of CSU contacts in the molecule
    stabilisers: sparse or dense array
        nresidues x nresidues array of CSU stabilising contacts in the molecule
    destabilisers: sparse or dense array
        nresidues x nresidues array of CSU destabilising contacts in the molecule
    res_idx: list
        list of serial residue ids for each of the residues
    G: nx.Graph
        residue based graph of the molecule
    '''
    if not sp.issparse(overlaps):
        overlaps = sp.csr_matrix(overlaps)
    if not sp.issparse(contacts):
        contacts = sp.csr_matrix(contacts)
    if not sp.issparse(stabilisers):
        stabilisers = sp.csr_matrix(stabilisers)
    if not sp.issparse(destabilisers):
        destabilisers = sp.csr_matrix(destabilisers)

    res_idx_inv = {int(v): i for i, v in enumerate(res_idx)}

    # Find active (i, j) pairs: union of nonzero positions in overlaps and contacts
    ov_coo = overlaps.tocoo()
    ct_coo = contacts.tocoo()
    all_i = np.concatenate([ov_coo.row, ct_coo.row])
    all_j = np.concatenate([ov_coo.col, ct_coo.col])

    contacts_list = []
    all_contacts = []

    if len(all_i) == 0:
        return contacts_list, all_contacts

    # Unique pairs in row-major order, excluding diagonal
    pairs = np.unique(np.column_stack([all_i, all_j]), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    for i1, i2 in pairs:
        over = overlaps[i1, i2]
        cont = contacts[i1, i2]
        stab = stabilisers[i1, i2]
        dest = destabilisers[i1, i2]
        rcsu = (stab - dest) > 0

        a = res_idx_inv[i1]
        b = res_idx_inv[i2]
        all_contacts.append([i1+1, i2+1, a, b, over, cont, stab, rcsu])
        if over == 1 or (over == 0 and rcsu):
            # this is a OV or rCSU contact we take it
            contacts_list.append((int(G.nodes[a]['stash']['resid']), G.nodes[a]['chain'],
                                  int(G.nodes[b]['stash']['resid']), G.nodes[b]['chain']))

    LOGGER.debug("Found {} Go contacts ({} total residue-residue interactions examined)",
                 len(contacts_list), len(all_contacts))
    return contacts_list, all_contacts


def _write_contacts(fout, all_contacts, ca_pos, G):
    '''
    write the contacts calculated to file
    fout: str
        path to write file to
    all_contacts: list
        list of lists of every contact found
    ca_pos: list
        list of (3,) arrays with the position of the CA atom of each residue
    G: nx.Graph
        residue graph of the input molecule
    '''

    header = [f"Go contact map calculated with vermouth {VERSION}\n\n"]

    header.append("Residue-Residue Contacts\n"
                  "\n"
                  "ID       - atom identification\n"
                  "I1,I2    - serial residue id\n"
                  "AA       - 3-letter code of aminoacid\n"
                  "C        - chain\n"
                  "I(PDB)   - residue number in PDB file\n"
                  "DCA      - distance between CA\n"
                  "CMs      - OV , CSU , oCSU , rCSU\n"
                  "           (CSU does not take into account chemical properties of atoms)\n"
                  "rCSU     - net contact from rCSU\n"
                  "Count    - number of contacts between residues\n"
                  "MODEL    - model number\n"
                  "\n"
                  "      ID    I1  AA  C I(PDB)     I2  AA  C I(PDB)        DCA       CMs    rCSU   Count  Model\n"
                  "=============================================================================================\n")

    msgs = []
    count = 0
    for contact in all_contacts:
        count += 1
        msg = (f"R {int(count):6d} "
               f"{int(contact[0]):5d}  {G.nodes[contact[2]]['resname']:3s} "
               f"{G.nodes[contact[2]]['chain']:1s} {int(G.nodes[contact[2]]['stash']['resid']):4d}    "
               f"{int(contact[1]):5d}  {G.nodes[contact[3]]['resname']:3s} "
               f"{G.nodes[contact[3]]['chain']:1s} {int(G.nodes[contact[3]]['stash']['resid']):4d}    "
               f"{euclidean(ca_pos[contact[2]], ca_pos[contact[3]])*10:9.4f}     "
               f"{int(contact[4]):1d} {1 if contact[5] != 0 else 0} "
               f"{1 if contact[6] != 0 else 0} {1 if contact[7] else 0}"
               f"{int(contact[7]): 6d}  {int(contact[5]): 6d}"
               f"     0\n")
        msgs.append(msg)
    message_out = ''.join(msgs)
    with deferred_open(fout, "w", encoding='utf-8') as f:
        f.write(''.join(header))
        f.write(message_out)


"""
Read RCSU Go model contact maps.
"""


def read_go_map(system, file_path):
    """
    Read a RCSU contact map from the c code as published in
    doi:10.5281/zenodo.3817447. The format requires all
    contacts to have 18 columns and the first column to be
    a capital R.

    Parameters
    ----------
    system: vermouth.system.System
        The system to process. Is modified in-place.
    file_path: :class:`pathlib.Path`
        path to the contact map file

    Returns
    -------
    list(tuple)
        contact as chain id, res id, chain id, res id
    """
    with open(file_path, "r", encoding='utf-8') as _file:
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

    system.go_params["go_map"].append(contacts)


def calculate_go_contacts(molecule, write_file):
    '''
    master function to calculate Go contacts

    molecule: vermouth.Molecule
        molecule to calculate contacts for
    write_file: bool
        write the file of the contacts out
    '''
    vdw_list, atypes, coords, res_serial, resids, chains, resnames, res_idx, ca_pos, nresidues, mol_graph = _extract_contact_inputs(
        molecule)

    LOGGER.info("Calculating Go contacts for {} residues ({} atoms)", nresidues, len(coords))

    overlaps, contacts, stabilisers, destabilisers = _compute_residue_contacts(vdw_list,
                                                                        atypes,
                                                                        coords,
                                                                        res_serial,
                                                                        nresidues)

    contacts, all_contacts = _filter_rcsu_contacts(overlaps, contacts,
                                            stabilisers, destabilisers,
                                            res_idx, mol_graph)

    LOGGER.info("Contact map complete: {} Go contacts identified", len(contacts))

    if isinstance(write_file, (str, Path)):
        _write_contacts(write_file, all_contacts, ca_pos, mol_graph)

    return contacts


class GenerateContactMap(Processor):
    """
    Processor to generate the contact rCSU contact map for a protein from an atomistic structure
    """
    def __init__(self, write_file):
        self.write_file = write_file

    def run_molecule(self, molecule):
        """
        Process `system`.

        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        return calculate_go_contacts(molecule, self.write_file)

    def run_system(self, system):
        for molecule in system.molecules:
            contacts = self.run_molecule(molecule)
            system.go_params["go_map"].append(contacts)
