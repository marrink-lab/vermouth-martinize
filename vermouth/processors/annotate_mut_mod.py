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

import networkx as nx

from .processor import Processor
from ..log_helpers import StyleAdapter, get_logger
from ..utils import maxes
from ..graph_utils import make_residue_graph
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


def residue_matches(resspec, residue_graph, res_idx):
    """
    Returns True iff resspec desribes residue_graph.nodes[res_idx]. The
    'resname's nter and cter match match the residues with a degree of 1 and 
    with the lowest and highest residue numbers respectively.

    Parameters
    ----------
    resspec: dict
        Attributes that must be present in the residue node. 'resname' is
        treated specially as described above.
    residue_graph: networkx.Graph
        A graph with one node per residue.
    res_idx: collections.abc.Hashable
        A node index in residue_graph.

    Returns
    -------
    bool
        Whether resspec describes the node res_idx in residue_graph.
    """
    res_node = residue_graph.nodes[res_idx]
    residue = {key: res_node.get(key)
               for key in 'chain resid resname insertion_code'.split()}
    out = True
    if resspec.get('resname', '')[-3:] == 'ter':
        # Find all residues with degree 1: the ones with the lowest resid will
        # be cter, the ones with the highest resid nter.
        termini = [idx for idx in residue_graph if residue_graph.degree[idx] == 1]
        get_resid = lambda idx: residue_graph.nodes[idx].get('resid')
        # FIXME: Once residue_graph is a digraph we can do something much much
        #        more clever, addressing arbitrarily branched polymers and
        #        termini
        if resspec['resname'] == 'nter':
            return res_idx in maxes(termini, key=lambda x: -get_resid(x))
        elif resspec['resname'] == 'cter':
            return res_idx in maxes(termini, key=get_resid)
        else:
            raise KeyError("Don't know any terminus with name '{}'".format(resspec['resname']))

        out = out and residue_graph.degree[res_idx] == 1
        del resspec['resname']
    return out and _subdict(resspec, residue)


def _format_resname(res):
    """
    Provisional function that performs the opposite of parse_residue_spec.
    Poorly tested, use at own risk.
    """
    chain = res.get('chain', '')
    out = ''
    if chain:
        out += chain + '-'
    resname = res.get('resname')
    out += resname
    if resname and resname[-1].isdigit():
        out += '#'
    out += str(res.get('resid', ''))
    out += res.get('insertion_code', '')
    return out


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

    residue_graph = make_residue_graph(molecule)
    for res_idx in residue_graph:
        for mutmod, key, library in [(modifications, 'modification', molecule.force_field.modifications),
                                     (mutations, 'mutation', molecule.force_field.blocks)]:
            for resspec, mod in mutmod:
                if residue_matches(resspec, residue_graph, res_idx):
                    if mod not in library:
                        raise NameError('{} is not known as a {} for '
                                        'force field {}'
                                       ''.format(mod, key, molecule.force_field.name))
                    res = residue_graph.nodes[res_idx]
                    LOGGER.debug('Annotating {} with {} {}',
                                 _format_resname(res), key, mod)
                    for node_idx in res['graph']:
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
