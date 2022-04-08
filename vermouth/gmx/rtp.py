# Copyright 2020 University of Groningen
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
Handle the RTP format from Gromacs.
"""

from collections import defaultdict, namedtuple
from itertools import groupby
import lark

from .. import DATA_PATH
from ..molecule import Block, Link, Interaction
from ..utils import first_alpha


__all__ = ['read_rtp']


GRAMMAR_FILE = DATA_PATH / 'grammars' / 'gmx_rtp.lark'
with open(GRAMMAR_FILE) as file:
    GRAMMAR = file.read()


_BondedTypes = namedtuple(
    '_BondedTypes',
    'bonds angles dihedrals impropers all_dihedrals nrexcl HH14 remove_dih cmap exclusions'
)


class Transformer(lark.visitors.Transformer_InPlace):
    def INT(self, tok):
        return tok.update(value=int(tok))

    def SIGNED_NUMBER(self, tok):
        return tok.update(value=float(tok))

    def PARAMETER(self, tok):
        return tok.value

    def atom(self, tokens):
        tokens = (tok.value for tok in tokens)
        return dict(zip(['atomname', 'atype', 'charge', 'charge_group'], tokens))


class RTPInterpreter(lark.visitors.Interpreter):
    ATOMS_PER_INTERACTION = {'bonds': 2, 'angles': 3, 'dihedrals': 4, 'impropers': 4, 'cmap': 5, 'exclusions': 2}
    def __init__(self, *args, force_field=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bonded_types = []
        self.block = None
        self.force_field = force_field
        self.links = []

    def bondedtypes(self, tree):
        # Default taken from
        # 'src/gromacs/gmxpreprocess/resall.cpp::read_resall' in the Gromacs
        # source code.
        defaults = 1, 1, 1, 1, 0, 3, 1, 1
        read = [tok.value for tok in tree.children]
        self.bonded_types = _BondedTypes(*(read + list(defaults[len(read):])), cmap=1, exclusions=1)

    def _count_hydrogens(self, idxs):
        names = [self.block.nodes[idx].get('atomname') for idx in idxs]
        return len([name for name in names if first_alpha(name) == 'H'])

    def _finish_block(self):
        self.block.make_edges_from_interactions()

        # Add all possible angles
        for atoms in self.block.guess_angles():
            self.block.add_interaction('angles', atoms=atoms, parameters=[self.bonded_types.angles])

        # Generate missing dihedrals
        # As pdb2gmx generates all the possible dihedral angles by default,
        # RTP files are written assuming they will be generated. A RTP file
        # have some control over these dihedral angles through the bondedtypes
        # section.
        all_dihedrals = []
        for center, dihedrals in groupby(
                sorted(self.block.guess_dihedrals(), key=_dihedral_sorted_center),
                _dihedral_sorted_center):
            if _keep_dihedral(center, self.block, self.bonded_types):
                # See src/gromacs/gmxpreprocess/gen_add.cpp::dcomp in the
                # Gromacs source code (see version 2016.3 for instance).
                # We only keep the one with 1) fewest hydrogens, and 2) lowest
                # indices
                atoms = min(dihedrals, key=lambda idxs: (self._count_hydrogens(idxs), idxs))
                all_dihedrals.append(Interaction(atoms=atoms, parameters=[self.bonded_types.dihedrals], meta={}))
        self.block.interactions['dihedrals'] = sorted(
            self.block.interactions.get('dihedrals', []) + all_dihedrals,
            key=lambda inter: inter.atoms
        )

        # TODO: generate 14 pairs, exclusions, ...

        # Split off the links. We usually generate 3 links, one -, one +, and
        # possibly one +-. This is because of terminal residues.
        link_groups = defaultdict(set)
        for interactions in self.block.interactions.values():
            for interaction in interactions:
                orders = frozenset(self.block.nodes[idx]['atomname'][0]
                                   for idx in interaction.atoms if
                                   self.block.nodes[idx]['atomname'][0] in '+-')
                if orders:
                    link_groups[orders].update(interaction.atoms)

        for link_nodes in link_groups.values():
            tmp_link = self.block.subgraph(link_nodes)
            link = Link(tmp_link)
            link.interactions = tmp_link.interactions
            for n_idx in link.nodes:
                # Keep only desired node attributes
                node = link.nodes[n_idx]
                for attr in list(node.keys()):
                    if attr not in ('atomname', 'resname'):
                        del node[attr]
                if node['atomname'][0] in '+-':
                    order, node['atomname'] = node['atomname'][0], node['atomname'][1:]
                    node['order'] = 1 if order == '+' else -1
            self.links.append(link)

        # Remove the foreign nodes from the block
        for n_idx in list(self.block.nodes):
            if self.block.nodes[n_idx]['atomname'][0] in '+-':
                self.block.remove_node(n_idx)

    def residue(self, tree):
        name = tree.children[0]
        self.block = Block(name=name.value, nrexcl=self.bonded_types.nrexcl, force_field=self.force_field)
        self.visit_children(tree)  # Interpret the children, i.e. atoms and interactions
        self._finish_block()

        self.force_field.blocks[name] = self.block
        self.force_field.links.extend(self.links)
        self.links = []

    def atomsection(self, tree):
        self.block.add_nodes_from(enumerate(tree.children))
        for node_idx in self.block:
            self.block.nodes[node_idx]['resname'] = self.block.name

    def _interaction(self, tree, type_):
        num_atoms = self.ATOMS_PER_INTERACTION[type_]
        atoms, parameters = tree.children[:num_atoms], tree.children[num_atoms]
        params = parameters.children if parameters else [getattr(self.bonded_types, type_)]
        atids = []
        for atom in atoms:
            if atom.value[0] in ('+-'):
                # We need to add the "link" atom/node so we can find edges based on
                # the link interactions. We separate these atoms and interactions
                # into Links later.
                try:
                    atid = next(self.block.find_atoms(atomname=atom.value))
                except StopIteration:
                    self.block.add_node(len(self.block), atomname=atom.value)
                    atid = len(self.block) - 1
            else:
                try:
                    atid = next(self.block.find_atoms(atomname=atom.value))
                except StopIteration as err:
                    raise KeyError('No atom with name {} is found in residue {}'
                                   ''.format(atom.value, self.block.name)) from err
            atids.append(atid)
        self.block.add_interaction(type_, atoms=atids, parameters=params)

    def bonds(self, tree):
        for child in tree.children:
            self._interaction(child, "bonds")

    def angles(self, tree):
        for child in tree.children:
            self._interaction(child, "angles")

    def dihedrals(self, tree):
        for child in tree.children:
            self._interaction(child, "dihedrals")

    def impropers(self, tree):
        for child in tree.children:
            self._interaction(child, "impropers")

    def cmaps(self, tree):
        for child in tree.children:
            self._interaction(child, "cmap")

    def exclusions(self, tree):
        for child in tree.children:
            self._interaction(child, "exclusions")


def _keep_dihedral(center, block, bondedtypes):
    if (not bondedtypes.all_dihedrals) and block.has_dihedral_around(center):
        return False
    if bondedtypes.remove_dih and block.has_improper_around(center):
        return False
    return True


def _dihedral_sorted_center(atoms):
    #return sorted(atoms[1:-1])
    return atoms[1:-1]


def read_rtp(lines, force_field):
    """
    Read blocks and links from a Gromacs RTP file to populate a force field

    Parameters
    ----------
    lines: collections.abc.Iterator
        An iterator over the lines of a RTP file (e.g. a file handle, or a
        list of string).
    force_field: vermouth.forcefield.ForceField
        The force field to populate in place.

    Raises
    ------
    IOError
        Something in the file could not be parsed.
    """
    parser = lark.Lark(GRAMMAR, maybe_placeholders=True)
    parsed = Transformer().transform(parser.parse(lines.read()))

    visitor = RTPInterpreter(force_field=force_field)
    visitor.visit(parsed)
