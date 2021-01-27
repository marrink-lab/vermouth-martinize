#!/usr/bin/env python3
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
Provides a processor for building missing coordinates.
"""

import numpy as np
import networkx as nx

from .processor import Processor
from .. import selectors


def mindist(A, B):
    return ((A[:, None] - B[None])**2).sum(axis=2).min(axis=1)


def get_atoms_missing_coords(molecule):
    '''
    Determine particles without coordinates in molecule.
    '''
    return [
        ix for ix in molecule
        if not selectors.selector_has_position(molecule.nodes[ix])
    ]


def get_anchors(molecule, indices):
    '''
    Determine particles with known coordinates connected to
    particles with given indices.
    '''
    return {
        a for ix in indices for a in molecule[ix].keys()
        if not selectors.selector_has_position(molecule.nodes[a]) is not None
    }


def align_z(v):
    '''
    Generate rotation matrix aligning z-axis onto v (and vice-versa).
    '''

    # This routine is based on the notion that for any two
    # vectors, the alignment is a 180 degree rotation
    # around the resultant vector.

    w = v / (v ** 2).sum() ** 0.5

    if w[2] <= 1e-8 - 1:
        return -np.eye(3) # Mind the inversion ...

    w[2] += 1

    return (2 / (w**2).sum()) * w * w[:, None] - np.eye(3)


def conicring(n=24, distance=0.125, angle=109.4, axis=None):
    '''
    Create a ring of positions corresponding to a cone with given angle.
    An angle of 109.4 is suitable for ideal SP3 substituents.
    '''
    p = 2 * np.pi * np.arange(n) / n
    r = distance * np.sin(angle * np.pi / 360)
    z = distance * np.cos(angle * np.pi / 360)
    X = np.stack([r * np.cos(p), r * np.sin(p), z * np.ones(n)], axis=1)
    if axis is None:
        return X
    return X @ align_z(axis)


def double_cone(n=24, distance=0.125, angle=109.4, axis=None):
    '''Create positions for two consecutive (SP3) bonded particles'''
    P = conicring(n, distance, angle, axis)
    S = np.array([P @ align_z(x) + x for x in P])
    return P, S


def triple_cone(n=24, distance=0.125, angle=109.4):
    '''Create possible positions for three consecutive (SP3) bonded particles'''
    P, S = double_cone(n, distance, angle)
    T = np.array([[[P @ align_z(u - x) + u] for u in U] for x, U in zip(P, S)])
    return P, S, T


def _out(molecule, anchor, base, distance=0.125):
    '''
    Generate coordinates for atom bonded to anchor
    using resultant vector of known substituents.

    This approach works for placement of single unknown
    substituents on sp1, sp2 or sp3 atoms. It should also
    work for bipyramidal and other configurations.

     b1
       \
    b2--a-->X
       /  u
     b3

    X := position to determine
    a := position of anchoring particle
    B = [b1, b2, ...] := positions of base particles

    Parameters
    ----------

    Returns
    -------
        None
    '''
    a = molecule.nodes[anchor]['position']
    B = [ molecule.nodes[ix]['position'] for ix in base ]
    B = B - a
    u = -B.sum(axis=0)
    u *= distance / ((u ** 2).sum() ** 0.5)
    return a + u


def _chiral(molecule, anchor, base, distance=(0.125, 0.125)):
    '''
    Generate coordinates for atoms bonded to chiral center
    based on two known substituents.

    This approach uses the positions of two substituents to
    generate coordinates for one or two atoms connected to
    a chiral center. The two atoms are indicated as `up` and
    `down`, referring to the direction with respect to the
    cross-product of the two substituents. If these control
    atoms are not given explicitly in order, then the control
    atoms are inferred from the connections of the anchor with
    known coordinates. In that case, the specific chirality
    of the center (R or S) cannot be controlled.

    NOTE: This function should probably be able to determine
          the weights of the substituents to allow placement
          based on R or S chirality.

    Parameters
    ----------


    Returns
    -------
        None
    '''
    a = molecule.nodes[anchor]['position']
    B = [ molecule.nodes[ix]['position'] for ix in base ]
    B = B - a
    u = -0.5 * (B[0] + B[1])
    v = np.cross(B[0], B[1])
    # Set length of v to length to half distance between subs
    v = 0.5 * ((B[0] - B[1])**2).sum()**0.5 * v / (v ** 2).sum()**0.5
    # Normalization of vector sum
    n = 1 / ((u + v)**2).sum()**0.5

    rup, rdown = distance

    pup = a + rup * n * (u + v)
    pdown = a + rdown * n * (u - v)

    return pup, pdown


class Segment:
    '''Class for subgraphs of (missing) atoms with anchors'''
    def __init__(self, molecule, limbs):
        self.molecule = molecule
        self.anchors = limbs[0][0]
        self.bases = limbs[0][1]
        self.limbs = [ s[2] for s in limbs ]

    def __str__(self):
        return ':'.join([str(self.anchors), str(self.bases), str(self.limbs)])

    def extrude(self, distance=0.125, coords=None, contact=0.5):
        mol = self.molecule

        for i, (a, b) in enumerate(zip(self.anchors, self.bases)):
            target = [ k for k in mol.neighbors(a) if k not in b ]
            valence = len(target) + len(b)

            if valence > 2 and len(b) == valence - 1:
                # Trivial chiral/flat/bipyramidal/octahedral/...
                mol.nodes[target[0]]['position'] = _out(mol, anchor=a, base=b)
                for k in self.limbs:
                    k.discard(target[0])
                # Simply update this segment and return it
                self.bases[i] = [a]
                self.anchors[i] = target[0]
                return [self]

            if valence == 4 and len(b) == 2:
                # Trivial chiral - except for the actual chirality
                # a simple 'up' or 'down' flag would be helpful
                # amino acid side chain is 'up'
                up, down = _chiral(mol, anchor=a, base=b, up=target[0], down=target[1])
                mol.nodes[target[0]]['position'] = up
                mol.nodes[target[1]]['position'] = down
                # Simply update this segment and return it
                for k in self.limbs:
                    k.discard(target[0])
                self.bases[i] = [a]
                self.anchors[i] = target[0]
                return [self]

            if valence == 1:
                # Generate a sphere of points
                # Check for overlaps
                # Take a pick
                # Should be done last anyway
                # Best collected and stored
                continue

            # So a valence of 2 (missing 1), 3 (missing 2) or 4 (missing 3)
            # The cases 2(1), 3(2), 4(3) can be

            if len(b) == 1 and all(len(s) < 100 for s in self.limbs):
                ax = mol.nodes[a]['position']
                bx = mol.nodes[b[0]]['position']
                P = conicring(axis=ax-bx) + ax
                if coords is not None:
                    P[mindist(P, coords) < contact**2] = np.nan
                P = P.reshape((len(self.limbs), -1, 3))
                for px in P.transpose((1,0,2)):
                    if px.max() == np.nan:
                        continue
                    # Split the limbs to new segments and return those
                    #out = []
                    for t, pi in zip(target, px):
                        self.molecule.nodes[t]['position'] = pi
                    #    for k in self.limbs:
                    #        if len(k) > 1 and t in k:
                    #            k.discard(t)
                    #            for g in nx.connected_components(self.molecule.subgraph(k)):
                    #                out.append(
                    #                    Segment(
                    #                        self.molecule,
                    #                        [(t, [[a]], g)],
                    #                        self.coords,
                    #                        self.contact
                    #                    )
                    #                )
                    #  This commented-out stuff may be useful for speeding up
                    #  coordinate generation, because it avoids needing lookup
                    #  of missing coordinates in subsequent cycles.
                    #return out
                    return
                break
        return None


class GenerateCoordinates(Processor):
    """
    A processor for generating coordinates for atoms without.
    """
    def __init__(self, contact=0.5):
        super().__init__()
        self.contact = contact
        self.coordinates = None

    def run_system(self, system):
        """
        Process `system`.

        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """

        self.coordinates = np.stack(
            [
                a.get('position', (np.nan, np.nan, np.nan))
                for m in system.molecules
                for i, a in m.atoms
            ]
        )

        for molecule in system.molecules:
            self.run_molecule(molecule)

    def run_molecule(self, molecule):
        """
        Process a single molecule. Must be implemented by subclasses.

        Parameters
        ----------
        molecule: vermouth.molecule.Molecule
            The molecule to process.

        Returns
        -------
        vermouth.molecule.Molecule
            The provided molecule with complete coordinates
        """

        # Missing atoms - those without 'positions'
        missing = get_atoms_missing_coords(molecule)

        prev = len(molecule)
        while prev - len(missing) > 0:
            # Limbs: subgraphs of missing atoms with anchors and bases
            limbs = []
            for g in nx.connected_components(molecule.subgraph(missing)):
                anchors = list(get_anchors(molecule, g))
                bases = [ list(get_anchors(molecule, [a])) for a in anchors ]
                limbs.append((anchors, bases, g))
            limbs.sort()

            # Segment: subgraphs of limbs sharing anchors and bases
            segments = []
            prev = None
            for lim in limbs:
                if lim[0] != prev:
                    segments.append([])
                    prev = lim[0]
                segments[-1].append(lim)
            segments = [ Segment(molecule, limbs) for limbs in segments ]

            # Extrude segments one layer at a time
            while segments:
                s = segments.pop(0)
                r = s.extrude(coords=self.coordinates, contact=self.contact)
                if r is not None:
                    segments.extend(r)

            prev = len(missing)
            missing = get_missing_atoms(molecule)
            #print(len(missing))

        return molecule

