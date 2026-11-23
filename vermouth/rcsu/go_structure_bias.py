# Copyright 2023 University of Groningen
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
Obtain the structural bias for the Go model.
"""
import numpy as np
import networkx as nx
from ..graph_utils import make_residue_graph
from ..molecule import Interaction
from ..processors.processor import Processor
from ..selectors import filter_minimal, select_backbone
from ..gmx.topology import NonbondParam
from .go_utils import get_go_type_from_attributes
from ..log_helpers import StyleAdapter, get_logger
import sys
LOGGER = StyleAdapter(get_logger(__name__))


def select_go_contacts(contact_map, res_graph, find_residue, res_dist, cutoff_short, cutoff_long, backbone):
    """
    Select all contacts from the contact map that according to their distance
    and graph connectivity are eligible to form a Go bond. A contact is
    selected once it is found in both directions.

    Parameters
    ----------
    contact_map: list[(int, str, int, str)]
        list of contacts defined by the residue index and chain identifier
        of both residues
    res_graph: networkx.Graph
        residue graph, see :func:`vermouth.graph_utils.make_residue_graph`
    find_residue: collections.abc.Callable
        returns the node in `res_graph` for a chain identifier and residue
        index, or None if there is none
    res_dist: int
        if nodes are closer than res_dist along the residue graph they are
        ignored
    cutoff_short: float
        distances in nm smaller than this are ignored
    cutoff_long: float
        distances in nm larger than this are ignored
    backbone: str
        name of the backbone bead

    Yields
    ------
    tuple
        the residue nodes and backbone nodes of both residues, and the
        distance between the backbone nodes
    """
    # find all pairs of residues that are within bonded distance of
    # res_dist
    connected_pairs = dict(nx.all_pairs_shortest_path_length(res_graph, cutoff=res_dist))
    # eligible contacts found in one direction so far
    one_direction = set()
    bad_chains_warning = False
    for resid_a, chain_a, resid_b, chain_b in contact_map:
        # identify the contact in the residue graph based on
        # chain ID and resid
        res_a = find_residue(chain_a, resid_a)
        res_b = find_residue(chain_b, resid_b)
        if res_a is None or res_b is None:
            if not bad_chains_warning:
                LOGGER.warning("Mismatch between chain IDs in pdb and contact map. This probably means the "
                               "chain IDs are missing in the pdb and the contact map has all chains = Z.")
                bad_chains_warning = True
            continue
        # make sure that both residues are not connected
        # note: contacts should be symmetric so we only
        # check against one
        if res_b in connected_pairs[res_a]:
            continue
        # now we lookup the backbone nodes within the residue contact
        try:
            bb_a = next(filter_minimal(res_graph.nodes[res_a]['graph'], select_backbone, bb_atomname=backbone))
            bb_b = next(filter_minimal(res_graph.nodes[res_b]['graph'], select_backbone, bb_atomname=backbone))
        except StopIteration:
            LOGGER.warning(f'No backbone atoms with name "{backbone}" found in molecule. '
                           'Check -go-backbone argument if your forcefield does not use this name for '
                           'backbone bead atoms. Go model cannot be generated. Will exit now.')
            sys.exit(1)
        # compute the distance between bb-beads and verify that it
        # satisfies the cut-off criteria
        dist = np.linalg.norm(res_graph.nodes[res_a]['graph'].nodes[bb_a]['position'] -
                              res_graph.nodes[res_b]['graph'].nodes[bb_b]['position'])
        if not cutoff_long > dist > cutoff_short:
            continue
        # select the contact if it has been found in the other direction
        # before; else, keep it and continue searching
        if (res_b, res_a) in one_direction:
            yield res_a, res_b, bb_a, bb_b, dist
        else:
            one_direction.add((res_a, res_b))


class ComputeStructuralGoBias(Processor):
    """
    Generate the Go model structural bias for a system
    of molecules. This processor class has two main
    functions: .contact_selector and .compute_bias.
    The .run_molecule function simply loops over all
    molecules in the system and calls the other two
    functions. The computed structural bias parameters
    are stored in `system.gmx_topology_params` and can
    be written out using the `vermouth.gmx.write_topology`
    function.

    Subclassing
    -----------
    In order to customize the Go-model structural bias
    it is recommended to subclass this function and
    overwrite the ``contact_selector`` method and/or
    the ``compute_bias`` method. This subclassed Processor
    then has to be added to the into the martinize2
    pipeline in place of the StructuralBiasWriter or as
    replacement in the GoPipeline.
    """
    def __init__(self,
                 cutoff_short,
                 cutoff_long,
                 go_eps,
                 res_dist,
                 moltype,
                 go_anchor_bead,
                 system=None,
                 res_graph=None):
        """
        Initialize the Processor with arguments required
        to setup the Go model structural bias.

        Parameters
        ----------
        contact_map: list[(str, int, str, int)]
            list of contacts defined as by the chain
            identifier and residue index
        cutoff_short: float
            distances in nm smaller than this are ignored
        cutoff_long: float
            distances in nm larger than this are ignored
        go_eps: float
            epsilon value of the structural bias in
            kJ/mol
        res_dist: int
            if nodes are closer than res_dist along
            the residue graph they are ignored; this
            is similar to sequence distance but takes
            into account disulfide bridges for example
        moltype: str
            name of the molecule to treat
        res_graph: :class:`vermouth.molecule.Molecule`
            residue graph of the molecule; if None it
            gets generated automatically
        system: :class:`vermouth.system.System`
            the system
        magic_number: float
            magic number for Go contacts from the old
            GoVirt script.
        backbone: str
            name of backbone atom where virtual site is placed
        """
        self.cutoff_short = cutoff_short
        self.cutoff_long = cutoff_long
        self.go_eps = go_eps
        self.res_dist = res_dist
        self.moltype = moltype
        self.backbone = go_anchor_bead
        # don't modify
        self.res_graph = None
        self.system = system
        self.__chain_id_to_resnode = {}
        self.conversion_factor = 2**(1/6)

    # do not overwrite when subclassing
    def _chain_id_to_resnode(self, chain, resid):
        """
        Return the node corresponding to the chain and
        resid. First time the function is run the dict
        is being created.

        Parameters
        ----------
        chain: str
            chain identifier
        resid: int
            residue index

        Returns
        -------
        dict
            a dict matching the chain,resid to the self.res_graph node
        """

        if self.__chain_id_to_resnode:
            if self.__chain_id_to_resnode.get((chain, resid), None) is not None:
                return self.__chain_id_to_resnode[(chain, resid)]
            else:
                LOGGER.debug(stacklevel=5, msg='chain-resid pair not found in molecule')
        # make sure we only generate self.__chain_id_to_resnode only once
        else:
            # for each residue collect the chain and residue in a dict
            # we use this later for identifying the residues from the
            # contact map
            for resnode in self.res_graph.nodes:
                chain_key = self.res_graph.nodes[resnode].get('chain', None)
                # in vermouth within a molecule all resid are unique
                # for a merged system the original resid of each node is preserved in the stash
                resid_key = self.res_graph.nodes[resnode]['stash'].get('resid')
                self.__chain_id_to_resnode[(chain_key, resid_key)] = resnode

            if (chain, resid) in self.__chain_id_to_resnode:
                return self.__chain_id_to_resnode[(chain, resid)]
            else:
                LOGGER.debug(stacklevel=5, msg='chain-resid pair not found in molecule')


    def contact_selector(self, molecule):
        """
        Select all contacts from the contact map
        that according to their distance and graph
        connectivity are eligible to form a Go
        bond and create exclusions between the
        backbone beads of those contacts.

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`

        Returns
        -------
        list[(collections.abc.Hashable, collections.abc.Hashable, float)]
            list of node keys and distance
        """
        # distance_matrix of eligible symmetrical pairs as tuple(node, node, dist)
        symmetrical_matrix = []
        contacts = select_go_contacts(self.system.go_params["go_map"][0], self.res_graph,
                                      self._chain_id_to_resnode, self.res_dist,
                                      self.cutoff_short, self.cutoff_long, self.backbone)
        for resA, resB, bb_node_A, bb_node_B, dist in contacts:
            atype_a = next(get_go_type_from_attributes(self.res_graph.nodes[resA]['graph'],
                                                       stash=self.res_graph.nodes[resA]['stash'],
                                                       chain=self.res_graph.nodes[resA].get('chain'),
                                                       prefix=self.moltype))
            atype_b = next(get_go_type_from_attributes(self.res_graph.nodes[resB]['graph'],
                                                       stash=self.res_graph.nodes[resB]['stash'],
                                                       chain=self.res_graph.nodes[resB].get('chain'),
                                                       prefix=self.moltype))
            # generate backbone-backbone exclusions
            # perhaps one day can be its own function
            excl = Interaction(atoms=(bb_node_A, bb_node_B),
                               parameters=[], meta={"group": "Go model exclusion"})
            molecule.interactions['exclusions'].append(excl)
            symmetrical_matrix.append((atype_a, atype_b, dist))
        return symmetrical_matrix

    def compute_go_interaction(self, contacts):
        """
        Compute the epsilon value given a distance between
        two nodes, figure out the atomtype name and store
        it in the systems attribute gmx_topology_params.

        Parameters
        ----------
        contacts: list[(str, str, float)]
            list of node-keys and their distance

        Returns
        ----------
        dict[frozenset(str, str): float]
            dict of interaction parameters indexed by atomtype
        """
        go_inters = {}
        for atype_a, atype_b, dist in contacts:
            sigma = dist / self.conversion_factor
            # find the go virtual-sites for this residue
            # probably can be done smarter but mehhhh
            contact_bias = NonbondParam(atoms=(atype_a, atype_b),
                                        sigma=sigma,
                                        epsilon=self.go_eps,
                                        meta={"comment": [f"go bond {dist}"]})
            self.system.gmx_topology_params["nonbond_params"].append(contact_bias)

    def run_molecule(self, molecule):
        self.res_graph = make_residue_graph(molecule)
        # compute the contacts; this also creates
        # the exclusions
        contacts = self.contact_selector(molecule)
        # compute the interaction parameters
        self.compute_go_interaction(contacts)
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
