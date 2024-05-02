#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""
Provides processors that can add and remove IDR specific bonds
"""

import functools

from .processor import Processor
from ..graph_utils import make_residue_graph
from ..rcsu.go_utils import get_go_type_from_attributes, _in_resid_region
import numpy as np
from ..selectors import select_backbone
from ..log_helpers import StyleAdapter, get_logger
from ..dssp import annotate_residues_from_sequence
LOGGER = StyleAdapter(get_logger(__name__))


def make_disorder_string(molecule, idr_regions):
    """
    make a string for annotating order and disorder along a molecule
    """

    order_disorder_seq = []

    for key, node in molecule.nodes.items():
        if select_backbone(node):
            _old_resid = node['_old_resid']
            if _in_resid_region(_old_resid, idr_regions):
                order_disorder_seq.append("D")  # D for disordered
            else:
                order_disorder_seq.append("F")  # F for folded

    order_disorder = ''.join(order_disorder_seq)

    return order_disorder


class IDRBonds(Processor):
    """
    Processor which adds additional bonded potentials, and removes
    unnecessary present ones to IDR regions in the protein.

    This processor is designed for the work described in the reference
    M3_GO, but is generally applicable for such circumstances where extra
    addition/removals are necessary.

    """

    def __init__(self, idr_regions = None):
        """
        Parameters
        ----------

        idr_regions:
            regions defining the IDRs
        """
        self.idr_regions = idr_regions

    def remove_cross_nb_interactions(self, molecule, res_graph):
        """
        Remove existing bonded interactions in idr region
        """

        #list of all the Go pairs in the molecule
        all_go_pairs = np.array([list(i.atoms) for i in self.system.gmx_topology_params["nonbond_params"] if 'W' not in list(i.atoms)])
        # list to record which items we don't want. cross = go potential between folded and disordered domain.
        all_cross_pairs = []

        for res_node in res_graph.nodes:
            resid = res_graph.nodes[res_node]['resid']
            _old_resid = res_graph.nodes[res_node]['_old_resid']
            chain = res_graph.nodes[res_node]['chain']

            if _in_resid_region(_old_resid, self.idr_regions):
                vs_go_node = next(get_go_type_from_attributes(res_graph.nodes[res_node]['graph'],
                                                              resid=resid,
                                                              chain=chain,
                                                              prefix=molecule.meta.get('moltype')))
                all_cross_pairs.append(np.where(all_go_pairs == vs_go_node)[0]) #just need the first one

        # make sure we only have one entry in case a site has more than one interaction
        all_cross_pairs = np.unique([x for xs in all_cross_pairs for x in xs])
        # delete the folded-disordered Go interactions from the list going backwards.
        # otherwise list order gets messed up.
        for i in reversed(all_cross_pairs):
            del self.system.gmx_topology_params["nonbond_params"][i]



    def run_molecule(self, molecule):
        """
        Assign water bias for a single molecule
        """
        if not self.system:
            raise IOError('This processor requires a system.')

        if not molecule.meta.get('moltype'):
            raise ValueError('The molecule does not have a moltype name.')

        if hasattr(molecule, 'res_graph'):
            res_graph = molecule.res_graph
        else:
            res_graph = make_residue_graph(molecule)
            molecule.res_graph = res_graph

        self.remove_cross_nb_interactions(molecule, res_graph)

        order_disorder = make_disorder_string(molecule, self.idr_regions)
        annotate_residues_from_sequence(molecule, "cgidr", order_disorder)

        return molecule

    def run_system(self, system):
        """
        Assign the water bias of the Go model to file. Biasing
        is always molecule specific i.e. no two different
        vermouth molecules can have the same bias.

        Parameters
        ----------
        system: :class:`vermouth.system.System`
        """
        if not self.idr_regions:
            return system
        self.system = system
        LOGGER.info("Applying extra bonded potentials to IDRs", type="step")
        super().run_system(system)