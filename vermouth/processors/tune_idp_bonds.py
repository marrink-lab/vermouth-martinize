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
Provides processors that can add and remove IDR specific bonds
"""

import functools

from .processor import Processor
from ..graph_utils import make_residue_graph
from ..rcsu.go_utils import get_go_type_from_attributes, _get_resid_region, _in_resid_region
import numpy as np
from vermouth import selectors
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


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
        for i in all_cross_pairs[::-1]:
            del self.system.gmx_topology_params["nonbond_params"][i]

    def get_idr_keys(self, molecule):
        """
        get the keys of BB and SC1 beads in IDR regions
        """

        BB = []
        SC1 = []

        for key, node in molecule.nodes.items():
            _old_resid = node['_old_resid']
            if _in_resid_region(_old_resid, self.idr_regions):
                region = _get_resid_region(_old_resid, self.idr_regions)
                if selectors.select_backbone(node):
                    BB.append(np.array([region, key]))
                    #if the residue is a GLY and doesn't have a sidechain we need to note it
                    if node.get('resname') == 'GLY':
                        SC1.append(np.array([region, np.nan]))
                if node.get('atomname') == 'SC1':
                    SC1.append(np.array([region, key]))
        return BB, SC1

    def add_idr_angles(self, molecule, BB, SC1):
        """
        add idr specific angles to the idr region
        """
        #find how many disordered regions we've been given
        regions = list(set([i[0] for i in BB]))

        # BB-BB-SC1 angles
        for region in regions:
            BB_keys = [i[1] for i in BB if i[0] == region]
            SC_keys = [i[1] for i in SC1 if i[0] == region]

            window = 2
            for i in range(len(BB_keys) - window + 1):
                BB_atoms = BB_keys[i: i + window]
                SC_atoms = SC_keys[i: i + window]
                #ie. if we don't have any residues without sidechains
                if all(~np.isnan(np.array(SC_atoms))):
                    bbs_atomlist = [BB_atoms[0], BB_atoms[1], SC_atoms[1]]
                    sbb_atomlist = [SC_atoms[0], BB_atoms[0], BB_atoms[1]]

                    molecule.add_or_replace_interaction('angles',
                                                        atoms=bbs_atomlist,
                                                        parameters=['10', '85', '10'],
                                                        meta={"group": "idp-fix", "comment": "BB-BB-SC1-v1",
                                                              "version": 1}
                                                        )

                    molecule.add_or_replace_interaction('angles',
                                                        atoms=sbb_atomlist,
                                                        parameters=['10', '85', '10'],
                                                        meta={"group": "idp-fix", "comment": "SC1-BB-BB-v1",
                                                              "version": 1}
                                                        )
        #SC1-BB-BB(GLY)-BB and BB-BB(GLY)-BB-SC1 dihedrals
        for region in regions:
            BB_keys = [i[1] for i in BB if i[0] == region]
            SC_keys = [i[1] for i in SC1 if i[0] == region]

            inds = np.where(np.isnan(SC_keys) == True)[0]

            for i in inds:
                sbb_atomlist = [SC_keys[i-1],
                                 BB_keys[i-1],
                                 BB_keys[i]]

                bbs_atomlist = [BB_keys[i],
                                 BB_keys[i+1],
                                 SC_keys[i+1]]
                try:
                    molecule.remove_interaction('angles',
                                                atoms=sbb_atomlist)
                except KeyError:
                    pass
                try:
                    molecule.remove_interaction('angles',
                                                atoms=bbs_atomlist)
                except KeyError:
                    pass
                molecule.add_or_replace_interaction('angles',
                                                    atoms=bbs_atomlist,
                                                    parameters=['10', '85', '10', ],
                                                    meta={"group": "idp-fix", "comment": "BB(GLY)-BB-SC1-v1",
                                                          "version": 1}
                                                    )

                molecule.add_or_replace_interaction('angles',
                                                    atoms=sbb_atomlist,
                                                    parameters=['10', '85', '10', ],
                                                    meta={"group": "idp-fix", "comment": "SC1-BB-BB(GLY)-v1",
                                                          "version": 1}
                                                    )

    def add_idr_dih(self, molecule, BB, SC1):
        """
        add idr specific dihedrals to the idr region
        """

        #find how many disordered regions we've been given
        regions = list(set([i[0] for i in BB]))

        # BB-BB-BB-BB dihedrals
        #make sure we're only applying interactions within regions
        for region in regions:
            keys = [i[1] for i in BB if i[0] == region]
            window = 4
            for i in range(len(keys) - window + 1):
                atoms = keys[i: i + window]
                try:
                    molecule.remove_interaction('dihedrals',
                                                atoms=atoms)
                except KeyError:
                    pass
                molecule.add_or_replace_interaction('dihedrals',
                                                    atoms = atoms,
                                                    parameters = ['9', '-120', '-1', '1'],
                                                    meta = {"group": "idp-fix", "comment": "BB-BB-BB-BB-v1", "version":1}
                                                    )
                molecule.add_or_replace_interaction('dihedrals',
                                                    atoms = atoms,
                                                    parameters = ['9', '-120', '-1', '2'],
                                                    meta={"group": "idp-fix", "comment": "BB-BB-BB-BB-v2", "version": 2}
                                                    )

        # SC1-BB-BB-SC1 dihedrals
        for region in regions:
            BB_keys = [i[1] for i in BB if i[0] == region]
            SC_keys = [i[1] for i in SC1 if i[0] == region]

            window = 2
            for i in range(len(BB_keys) - window + 1):
                BB_atoms = BB_keys[i: i + window]
                SC_atoms = SC_keys[i: i + window]
                #ie. if we don't have any residues without sidechains
                if all(~np.isnan(np.array(SC_atoms))):
                    atomlist = [SC_atoms[0], BB_atoms[0], BB_atoms[1], SC_atoms[1]]
                    try:
                        molecule.remove_interaction('dihedrals',
                                                    atoms=atomlist)
                    except KeyError:
                        pass

                    molecule.add_or_replace_interaction('dihedrals',
                                                        atoms=atomlist,
                                                        parameters=['9', '-130', '-1.5', '1'],
                                                        meta={"group": "idp-fix", "comment": "SC1-BB-BB-SC1-v1",
                                                              "version": 1}
                                                        )
                    molecule.add_or_replace_interaction('dihedrals',
                                                        atoms=atomlist,
                                                        parameters=['9', '100', '-1.5', '2'],
                                                        meta={"group": "idp-fix", "comment": "SC1-BB-BB-SC1-v2",
                                                              "version": 2}
                                                        )
        #SC1-BB-BB(GLY)-BB and BB-BB(GLY)-BB-SC1 dihedrals
        for region in regions:
            BB_keys = [i[1] for i in BB if i[0] == region]
            SC_keys = [i[1] for i in SC1 if i[0] == region]

            inds = np.where(np.isnan(SC_keys) == True)[0]

            for i in inds:
                sbbb_atomlist = [SC_keys[i-1],
                                 BB_keys[i-1],
                                 BB_keys[i],
                                 BB_keys[i+1]]

                bbbs_atomlist = [BB_keys[i-1],
                                 BB_keys[i],
                                 BB_keys[i+1],
                                 SC_keys[i+1]]

                try:
                    molecule.remove_interaction('dihedrals',
                                                atoms=sbbb_atomlist)
                except KeyError:
                    pass
                try:
                    molecule.remove_interaction('dihedrals',
                                            atoms=bbbs_atomlist)
                except KeyError:
                    pass
                molecule.add_or_replace_interaction('dihedrals',
                                                    atoms=sbbb_atomlist,
                                                    parameters=['1', '115', '-4.5', '1'],
                                                    meta={"group": "idp-fix", "comment": "SC1-BB-BB(GLY)-BB-v1",
                                                          "version": 1}
                                                    )
                molecule.add_or_replace_interaction('dihedrals',
                                                    atoms=bbbs_atomlist,
                                                    parameters=['1', '0', '-2.0', '1'],
                                                    meta={"group": "idp-fix", "comment": "BB-BB(GLY)-BB-SC1-v1",
                                                          "version": 1}
                                                    )

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

        self.remove_cross_nb_interactions(molecule, res_graph)

        BB, SC1 = self.get_idr_keys(molecule)
        self.add_idr_dih(molecule, BB, SC1)
        self.add_idr_angles(molecule, BB, SC1)

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
