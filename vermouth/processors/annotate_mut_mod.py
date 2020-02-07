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
Provides a processor that annotates a molecule with desired mutations and
modifications.
"""

from .processor import Processor
from ..log_helpers import StyleAdapter, get_logger
from ..utils import format_atom_string
from ..graph_utils import collect_residues
LOGGER = StyleAdapter(get_logger(__name__))


def parse_residue_spec(resspec):
    """
    Parse a residue specification: [<chain>-][<resname>][[#]<resid>] where 
    resid is /[0-9]+/.
    If resname ends in a number and a resid is also specified, the # separator
    is required.
    Returns a dictionary with keys 'chain', 'resname', and 'resid' for the
    fields that are specified. Resid will be an int.

    Parameters
    ----------
    resspec: str

    Returns
    -------
    dict
    """
    # A-LYS2 or PO4#2
    # <chain>-<resname><resid>
    *chain, res = resspec.split('-', 1)
    res, *resid = res.rsplit('#', 1)
    if resid:  # [] if False
        resname = res
        resid = resid[0]
    else:
        idx = 0
        for idx, char in reversed(list(enumerate(res))):
            if not char.isdigit():
                idx += 1
                break
        resname = res[:idx]
        resid = res[idx:]

    out = {}
    if resid:
        resid = int(resid)
        out['resid'] = resid
    if resname:
        out['resname'] = resname
    if chain:
        out['chain'] = chain[0]
    return out


def _subdict(dict1, dict2):
    """True if dict1 <= dict2
    All items in dict1 must be in dict2.
    """
    for key, val in dict1.items():
        if key not in dict2 or dict2[key] != val:
            return False
    return True


def annotate_modifications(molecule, modifications, mutations):
    """
    Annotate nodes in molecule with the desired modifications and mutations

    Parameters
    ----------
    molecule: networkx.Graph
    modifications: list[tuple[dict, str]]
        The modifications to apply. The first element is a dictionary contain
        the attributes a residue has to fulfill. It can contain the elements
        'chain', 'resname' and 'resid'. The second element is the modification
        that should be applied.
    mutations: list[tuple[dict, str]]
        The mutations to apply. The first element is a dictionary contain
        the attributes a residue has to fulfill. It can contain the elements
        'chain', 'resname' and 'resid'. The second element is the mutation that
        should be applied.

    Raises
    ------
    NameError
        When a modification is not recognized.
    """
    if not modifications and not mutations:
        return

    residues = collect_residues(molecule)
    for residue, node_idxs in residues.items():
        residue = dict(zip(('chain', 'resid', 'resname'), residue))
        for mutmod, key, library in [(modifications, 'modification', molecule.force_field.modifications),
                                     (mutations, 'mutation', molecule.force_field.blocks)]:
            for resspec, mod in mutmod:
                if _subdict(resspec, residue):
                    if mod not in library:
                        raise NameError('{} is not known as a {} for '
                                        'force field {}'
                                       ''.format(mod, key, molecule.force_field.name))
                    for node_idx in node_idxs:
                        molecule.nodes[node_idx][key] = molecule.nodes[node_idx].get(key, []) + [mod]


class AnnotateMutMod(Processor):
    def __init__(self, modifications=None, mutations=None):
        if not modifications:
            modifications = []
        if not mutations:
            mutations = []
        self.modifications = []
        for resspec, val in modifications:
            self.modifications.append((parse_residue_spec(resspec), val))
        self.mutations = []
        for resspec, val in mutations:
            self.mutations.append((parse_residue_spec(resspec), val))

    def run_molecule(self, molecule):
        annotate_modifications(molecule, self.modifications, self.mutations)
        return molecule
