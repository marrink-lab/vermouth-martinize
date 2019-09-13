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
Quick-fix processor to *temporarily* fix the -nt option. Aims to make termini
neutral again. Should be replaced by a more permanent fix for #5.
"""

from .processor import Processor


NEUTRAL_EQUIV = {('C-ter', ): 'COOH-ter', ('N-ter', ): 'NH2-ter'}


def neutral_termini(molecule, modifications):
    """
    Fragile solution that makes termini neutral again, based on `NEUTRAL_EQUIV`
    and the known modifications.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule to work on. Relevant nodes are expected to have a
        `modification` attribute, usually set by the
        :class:`vermouth.processors.do_mapping.DoMapping`.
    modifications: dict[str, vermouth.molecule.Link]
        Modifications known in the force field.

    Note
    ----
    `molecule` is modified in-place!

    Note
    ----
    Atom names between the existing modifications and new modifications *must*
    match. If this is not the case, this is not detected. In addition, for best
    (read: correct) results the number of atoms in both modifications must be
    the same.

    Returns
    -------
    vermouth.molecule.Molecule
        The molecule with neutral termini.
    """
    for mol_idx in molecule:
        mol_node = molecule.nodes[mol_idx]
        if 'modification' in mol_node and mol_node['modification'].name in NEUTRAL_EQUIV:
            current_mod = mol_node['modification']
            new_mod = modifications[NEUTRAL_EQUIV[current_mod.name]]
            for mod_idx in new_mod:
                if new_mod.nodes[mod_idx]['atomname'] == mol_node['atomname']:
                    mol_node.update(new_mod.nodes[mod_idx].get('replace', {}))

    return molecule


class NeutralTermini(Processor):
    def __init__(self, force_field):
        self.ff = force_field

    def run_molecule(self, molecule):
        return neutral_termini(molecule, self.ff.modifications)
