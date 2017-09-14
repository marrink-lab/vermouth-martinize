# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:58:04 2017

@author: Peter Kroon
"""


import networkx as nx


class Molecule(nx.Graph):
#    angle_factory = dict
#    angle_attr_dict_factory = dict
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atoms = self.nodes
        self.bonds = self.edges
        self._angles = self.adjlist_dict_factory()
        self._dihedrals = self.adjlist_dict_factory()
    
    @property
    def angle(self):
        return self._angles
    
    @property
    def dihedral(self):
        return self._dihedrals
    
    def angles(self, data=False):
        return list(self.angles_iter(data))
    
    def dihedrals(self, data=False):
        return list(self.dihedrals_iter(data))
    
    def _deep_add(self, source, idxs, attr):
        deepest = []
        for idxs in (idxs, tuple(reversed(idxs))):
            data = source
            for n in idxs[:-1]:
                if n not in data:
                    data[n] = self.adjlist_dict_factory()
                data = data[n]
            deepest.append(data)
        datadict = data.get(idxs[-1], self.edge_attr_dict_factory())
        datadict.update(attr)
        # Why it's not idxs[-1], idxs[0] is a mystery
        for data, idx in zip(deepest, (idxs[0], idxs[-1])):
            data[idx] = datadict
    
    def add_angle(self, u, v, w, **attr):
        if not self.has_edge(u, v):
            self.add_edge(u, v)
        if not self.has_edge(v, w):
            self.add_edge(v, w)
        if not self.has_angle(u, v, w):
            self._deep_add(self._angles, (u, v, w), attr)
        else:
            self._angles[u][v][w].update(attr)
    
    def add_dihedral(self, u, v, w, x, **attr):
        for n in (u, v, w, x):
            if not self.has_node(n):
                self.add_node(n)
        # Add bonds?
        if not self.has_dihedral(u, v, w, x):
            self._deep_add(self._dihedrals, (u, v, w, x), attr)
        else:
            self._dihedrals[u][v][w][x].update(attr)
    
    def has_dihedral(self, u, v, w, x):
        try:
            return x in self._dihedrals[u][v][w]
        except KeyError:
            return False

    def has_angle(self, u, v, w):
        try:
            return w in self._angles[u][v]
        except KeyError:
            return False
    
    @classmethod
    def _rec_iter(cls, mapping, depth, item=None):
        if item is None:
            item = []
        if not mapping:
            yield tuple(item)
        for k, v in mapping.items():
            if depth == 0:
                yield tuple(item)
            else:
                yield from cls._rec_iter(v, depth-1, item + [k])
    
    @classmethod
    def _deep_iter(cls, source, depth, data):
        seen = set()
        for idxs in cls._rec_iter(source, depth):
            if idxs in seen or idxs[::-1] in seen:
                continue
            seen.add(idxs)
            if not data:
                yield idxs
            else:
                out = []
                data = source
                for n in idxs:
                    out.append(n)
                    data = data[n]
                out.append(data)
                yield tuple(out)

    def angles_iter(self, data=False):
        yield from self._deep_iter(self._angles, 3, data)
    
    def dihedrals_iter(self, data=False):
        yield from self._deep_iter(self._dihedrals, 4, data)
    
    def lookup(self, atomname, resid):
        for node_idx, node in self.node:
            if node['atomname'] == atomname and node['resid'] == resid:
                return node_idx
        raise KeyError('Can not find atom {} in residue number {}'.format(atomname, resid))


m = Molecule()
m.add_angle(0, 1, 2, bar='foo')
print(m.angle)
m.angle[0][1][2]['foo'] = 'bar'
print(m.angle)
print(list(m.angles_iter(data=True)))
m.add_dihedral(1, 2, 3, 4)
print(list(m.dihedrals_iter()))