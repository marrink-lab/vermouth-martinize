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
"""
import networkx as nx
from collections import defaultdict
from ..molecule import Interaction
from ..processors.processor import Processor
from ..gmx.topology import Atomtype
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


class VirtualSiteCreator(Processor):
    """
    Create virtual-sites for the Martini Go model implementation or
    the specific water biasing options.

    See :mod:`vermouth.rcsu` for more details.

    Every molecule must have a moltype name under the "moltype" key of the
    molecule meta.

    See Also
    --------
    :class:`~vermouth.processors.name_moltype.NameMolType`
        Assign molecule type names to the molecules in a system.
    :func:`add_virtual_sites`
    """
    def __init__(self, go_anchor_bead='BB', go_atomname='CA'):
        self.system = None
        self.backbone = go_anchor_bead
        self.atomname = go_atomname

    def run_molecule(self, molecule):
        moltype = molecule.meta.get('moltype')
        if not moltype:
            raise ValueError('The molecule does not have a moltype name.')

        if not self.system:
            raise ValueError('This processor requires a system.')
        # add citations for the go model here
        molecule.citations.add('M3_GO')

        self.add_virtual_sites(molecule,
                               prefix=moltype,
                               backbone=self.backbone,
                               atomname=self.atomname)

        return molecule

    def run_system(self, system):
        self.system = system
        LOGGER.info("Adding Virtual Sites to backbone beads.", type="step")
        super().run_system(system)

    def add_virtual_sites(self, molecule, prefix, backbone, atomname, charge=0):
        """
        Add the virtual sites for GoMartini in the molecule.

        One virtual site is added per backbone bead of the Martini protein.
        Each virtual site copies the resid, resname, and chain of the backbone
        bead. It also copies the *reference* to the position array, so the virtual
        site position follows if the backbone bead is translated. The virtual sites
        are added *after* all the other atoms of the molecule, each in its own
        charge group, with "CA" as atomname, and a charge of 0. The atomname and
        charge can be set with the `atomname` and `charge` argument, respectively.

        The bead type of the virtual sites is names "<prefix>_<resid>". Where
        `prefix` is provided as an argument of the function, and is expected to be
        the molecule type name.

        Parameters
        ----------
        molecule: vermouth.molecule.Molecule
            The molecule to augment with virtual sites.
        prefix: str
            The prefix to use for bead type names. Usually the molecule type name.
        backbone: str
            The atomname of the backbone beads.
        atomname: str
            The atomname of the virtual sites.
        charge: float or int
            The charge of the virtual sites.
        """
        # If there are no atoms, then there is nothing to do. We can exit early,
        # avoiding to deal with empty iterators.
        if not molecule.nodes:
            return
        virtual_site_nodes = []
        virtual_sites = []
        atomtypes = {}
        new_node_id = max(molecule.nodes)
        charge_groups = nx.get_node_attributes(molecule, 'charge_group').values()
        new_charge_group = max(charge_groups) if charge_groups else 0

        # Pre-Listing all beads that are built as VS, storing their idx & inter params 
        # i.e. bead is the first index in a virtual_sties* interaction.
        # Not a big fan of this.
        vs_first_interactions_by_node = defaultdict(list)
        for name, inters in molecule.interactions.items():
            if not name.startswith('virtual_sites'):
                continue
            for inter in inters:
                if inter.atoms:
                    vs_first_interactions_by_node[inter.atoms[0]].append((name, inter))

        # Collect new  VS interactions per VS type name so we can append
        # them to the correct lists afterwards.
        new_vs_by_name = defaultdict(list)

        for node_id, atom in molecule.nodes(data=True):
            if atom.get('atomname') == backbone:
                new_node_id += 1
                new_charge_group += 1

                virtual_site_nodes.append((new_node_id, {
                    'resid': atom['resid'],
                    'resname': atom['resname'],
                    'atype': '{}_{}'.format(prefix, atom['resid']),
                    'charge_group': new_charge_group,
                    'chain': atom['chain'],
                    'position': atom['position'],
                    'atomname': atomname,
                    'charge': charge,
                    'mass': 0.0,
                    'cgsecstruct': atom.get('cgsecstruct', None),
                    'stash': atom.get('stash', None)
                }))

                # If this backbone node is itself a VS (i.e.
                # it is the first atom in one or more virtual_sites* interactions),
                # clone those virtual-site interactions but replace the first atom
                # with the newly created Go virtual site node. 
                existing_vs_inters = vs_first_interactions_by_node.get(node_id)
                if existing_vs_inters:
                    for name, inter in existing_vs_inters:
                        cloned_atoms = [new_node_id] + list(inter.atoms[1:])
                        cloned_params = list(inter.parameters) if inter.parameters is not None else []
                        cloned_meta = dict(inter.meta) if inter.meta is not None else {}
                        cloned_meta['group'] = 'Virtual go site'
                        new_vs_by_name[name].append(Interaction(
                            atoms=cloned_atoms,
                            parameters=cloned_params,
                            meta=cloned_meta,
                        ))
                else:
                    # otherwise create a simple type 1 VS interaction as before
                    new_vs_by_name['virtual_sitesn'].append(Interaction(
                        atoms=[new_node_id, node_id],
                        parameters=['1'],
                        meta={'group': 'Virtual go site'},
                    ))

                vs_params = Atomtype(node=new_node_id, molecule=molecule, sigma=0.0, epsilon=0.0, meta={})
                self.system.gmx_topology_params['atomtypes'].append(vs_params)

        molecule.add_nodes_from(virtual_site_nodes)

        # append new interactions 
        for name, inters in new_vs_by_name.items():
            molecule.interactions[name] += inters