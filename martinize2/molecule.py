# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:58:04 2017

@author: Peter Kroon
"""

from collections import defaultdict
from functools import partial

import networkx as nx


class Molecule(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interactions = defaultdict(list)
    
    def copy(self, as_view=False):
        copy = super().copy(as_view)
        if not as_view:
            return self.__class__(copy)
        return copy

    def add_interaction(self, type_, atoms, parameters):
        for atom in atoms:
            if atom not in self:
                # KeyError?
                raise ValueError('Unknown atom {}'.format(atom))
        self.interactions[type_].append((tuple(atoms), parameters))

    def get_interaction(self, type_):
        return self.interactions[type_]

    def remove_interaction(self, type_, atoms):
        for idx, interaction in enumerate(self.interactions[type_]):
            if interaction[0] == atoms:
                break
        else:  # no break
            raise KeyError("Can't find interaction of type {} between atoms {}".format(type_, atoms))
        del self.interactions[type_][idx]

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
