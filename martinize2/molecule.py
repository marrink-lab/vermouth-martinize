# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:58:04 2017

@author: Peter Kroon
"""

from collections import defaultdict, OrderedDict, namedtuple
import copy
from functools import partial

import networkx as nx


Interaction = namedtuple('Interaction', 'atoms parameters meta')


class Molecule(nx.Graph):
    # As the particles are stored as nodes, we want the nodes to stay
    # ordered.
    node_dict_factory = OrderedDict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interactions = defaultdict(list)

    @property
    def atoms(self):
        for node in self.nodes():
            node_attr = self.node[node]
            yield node, node_attr

    def copy(self, as_view=False):
        copy = super().copy(as_view)
        if not as_view:
            return self.__class__(copy)
        return copy

    def subgraph(self, *args, **kwargs):
        return self.__class__(super().subgraph(*args, **kwargs))

    def add_interaction(self, type_, atoms, parameters, meta=None):
        if meta is None:
            meta = {}
        for atom in atoms:
            if atom not in self:
                # KeyError?
                raise ValueError('Unknown atom {}'.format(atom))
        self.interactions[type_].append(
            Interaction(atoms=tuple(atoms), parameters=parameters, meta=meta)
        )

    def get_interaction(self, type_):
        return self.interactions[type_]

    def remove_interaction(self, type_, atoms):
        for idx, interaction in enumerate(self.interactions[type_]):
            if interaction.atoms == atoms:
                break
        else:  # no break
            raise KeyError("Can't find interaction of type {} between atoms {}".format(type_, atoms))
        del self.interactions[type_][idx]

    def find_atoms(self, atomname, resname='', resid=-1):
        for node_idx in self:
            node = self.nodes[node_idx]
            if node['atomname'] == atomname and\
                    (not resname or node['resname'] == resname) and\
                    (resid == -1 or node['resid'] == resid):
                yield node_idx

    def __getattr__(self, name):
        # TODO: DRY
        if name.startswith('get_') and name.endswith('s'):
            type_ = name[len('get_'):-len('s')]
            return partial(self.get_interaction, type_)
        elif name.startswith('add_'):
            type_ = name[len('add_'):]
            return partial(self.add_interaction, type_)
        elif name.startswith('remove_'):
            type_ = name[len('remove_'):]
            return partial(self.remove_interaction, type_)
        else:
            raise AttributeError

    def merge_molecule(self, molecule):
        """
        Add the atoms and the interactions of a molecule at the end of this one.

        Atom and residue index of the new atoms are offset to follow the last
        atom of this molecule.
        """
        if len(self.nodes()):
            # We assume that the last id is always the largest.
            last_node_idx = max(self) 
            offset = last_node_idx + 1
            residue_offset = self.node[last_node_idx]['resid'] + 1
            offset_charge_group = self.node[last_node_idx].get('charge_group', -1) + 1
        else:
            offset = 0
            residue_offset = 0
            offset_charge_group = 0

        correspondence = {}
        for idx, node in enumerate(molecule.nodes(), start=offset):
            correspondence[node] = idx
            new_atom = copy.copy(molecule.node[node])
            new_atom['resid'] += residue_offset
            new_atom['charge_group'] = (new_atom.get('charge_group', 0)
                                        + offset_charge_group)
            self.add_node(idx, **new_atom)

        for name, interactions in molecule.interactions.items():
            for interaction in interactions:
                atoms = tuple(correspondence[atom] for atom in interaction.atoms)
                self.add_interaction(name, atoms, interaction.parameters)

        for edge in molecule.edges:
            self.add_edge(*(correspondence[node] for node in edge))


class Block(nx.Graph):
    """
    Residue topology template

    Attributes
    ----------
    name: str or None
        The name of the residue. Set to `None` if undefined.
    atoms: iterator of dict
        The atoms in the residue. Each atom is a dict with *a minima* a key
        'name' for the name of the atom, and a key 'atype' for the atom type.
        An atom can also have a key 'charge', 'charge_group', 'comment', or any
        arbitrary key. 
    interactions: dict
        All the known interactions. Each item of the dictionary is a type of
        interaction, with the key being the name of the kind of interaction
        using Gromacs itp/rtp conventions ('bonds', 'angles', ...) and the
        value being a list of all the interactions of that type in the residue.
        An interaction is a dict with a key 'atoms' under which is stored the
        list of the atoms involved (referred by their name), a key 'parameters'
        under which is stored an arbitrary list of non-atom parameters as
        written in a RTP file, and arbitrary keys to store custom metadata. A
        given interaction can have a comment under the key 'comment'.
    """
    # As the particles are stored as nodes, we want the nodes to stay
    # ordered.
    node_dict_factory = OrderedDict

    def __init__(self):
        super(Block, self).__init__(self)
        self.name = None
        self.interactions = {}

    def __repr__(self):
        name = self.name
        if name is None:
            name = 'Unnamed'
        return '<{} "{}" at 0x{:x}>'.format(self.__class__.__name__,
                                          name, id(self))

    def add_atom(self, atom):
        try:
            name = atom['atomname']
        except KeyError:
            raise ValueError('Atom has no atomname: "{}".'.format(atom))
        self.add_node(name, **atom)

    @property
    def atoms(self):
        for node in self.nodes():
            node_attr = self.node[node]
            # In pre-blocks, some nodes correspond to particles in neighboring
            # residues. These node do not carry particle information and should
            # not appear as particles.
            if node_attr:
                yield node_attr

    def make_edges_from_interaction_type(self, type_):
        """
        Create edges from the interactions of a given type.

        The interactions must be described so that two consecutive atoms in an
        interaction should be linked by an edge. This is the case for bonds,
        angles, proper dihedral angles, and cmap torsions. It is not always
        true for improper torsions.

        Cmap are described as two consecutive proper dihedral angles. The
        atoms for the interaction are the 4 atoms of the first dihedral angle
        followed by the next atom forming the second dihedral angle with the
        3 previous ones. Each pair of consecutive atoms generate an edge.

        .. warning::

            If there is no interaction of the required type, it will be
            silently ignored.

        Parameters
        ----------
        type_: str
            The name of the interaction type the edges should be built from.
        """
        for interaction in self.interactions.get(type_, []):
            atoms = interaction.atoms
            self.add_edges_from(zip(atoms[:-1], atoms[1:]))

    def make_edges_from_interactions(self):
        """
        Create edges from the interactions we know how to convert to edges.

        The known interactions are bonds, angles, proper dihedral angles, and
        cmap torsions.
        """
        known_types = ('bonds', 'angles', 'dihedrals', 'cmap')
        for type_ in known_types:
            self.make_edges_from_interaction_type(type_)

    def guess_angles(self):
        for a in self.nodes():
            for b in self.neighbors(a):
                for c in self.neighbors(b):
                    if c == a:
                        continue
                    yield (a, b, c)

    def guess_dihedrals(self, angles=None):
        angles = angles if angles is not None else self.guess_angles()
        for a, b, c in angles:
            for d in self.neighbors(c):
                if d not in (a, b):
                    yield (a, b, c, d)

    def has_dihedral_around(self, center):
        """
        Returns True if the block has a dihedral centered around the given bond.

        Parameters
        ----------
        center: tuple
            The name of the two central atoms of the dihedral angle. The
            method is sensitive to the order.

        Returns
        -------
        bool
        """
        all_centers = [tuple(dih['atoms'][1:-1])
                       for dih in self.interactions.get('dihedrals', [])]
        return tuple(center) in all_centers

    def has_improper_around(self, center):
        """
        Returns True if the block has an improper centered around the given bond.

        Parameters
        ----------
        center: tuple
            The name of the two central atoms of the improper torsion. The
            method is sensitive to the order.

        Returns
        -------
        bool
        """
        all_centers = [tuple(dih.atoms[1:-1])
                       for dih in self.interactions.get('impropers', [])]
        return tuple(center) in all_centers

    def to_molecule(self, atom_offset=0, resid=1, offset_charge_group=1):
        name_to_idx = {}
        mol = Molecule()
        for idx, atom in enumerate(self.atoms, start=atom_offset):
            name_to_idx[atom['atomname']] = idx
            new_atom = copy.copy(atom)
            new_atom['resid'] = resid
            new_atom['resname'] = self.name
            new_atom['charge_group'] = (new_atom.get('charge_group', 0)
                                        + offset_charge_group)
            mol.add_node(idx, **new_atom)
        for name, interactions in self.interactions.items():
            for interaction in interactions:
                atoms = tuple(
                    name_to_idx[atom] for atom in interaction.atoms
                )
                mol.add_interaction(
                    name, atoms,
                    interaction.parameters
                )
        for edge in self.edges:
            mol.add_edge(*(name_to_idx[node] for node in edge))

        return mol


class Link(nx.Graph):
    """
    Template link between two residues.
    """
    node_dict_factory = OrderedDict

    def __init__(self):
        super(Link, self).__init__(self)
        self.interactions = {}


if __name__ == '__main__':
    mol = Molecule()
    mol.add_edge(0, 1)
    mol.add_edge(1, 2)
    nx.subgraph(mol, (0, 1))

    mol.add_interaction('bond', (0, 1), tuple((1, 2)))
    mol.add_interaction('bond', (1, 2), tuple((10, 20)))
    mol.add_angle((0, 1, 2), tuple([10, 2, 3]))

    print(mol.interactions)
    print(mol.get_interaction('bond'))
    print(mol.get_bonds())
    print(mol.get_angles())

    mol.remove_interaction('bond', (0, 3))
    print(mol.get_bonds())
