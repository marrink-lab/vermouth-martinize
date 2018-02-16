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
Created on Wed Oct 25 16:00:02 2017

@author: peterkroon
"""
from ..molecule import LinkPredicate, attributes_match, NotDefinedOrNot
from .processor import Processor
from ..gmx import read_rtp

from itertools import combinations

import networkx as nx


class LinkGraphMatcher(nx.isomorphism.GraphMatcher):
    def semantic_feasibility(self, node1_name, node2_name):
        # TODO: implement (partial) wildcards
        # Node2 is the link
        node1 = self.G1.nodes[node1_name]
        node2 = self.G2.nodes[node2_name]
        return _atoms_match(node1, node2)


def _atoms_match(node1, node2):
    return attributes_match(node1, node2, ignore_keys=('order', 'replace'))


def _is_valid_non_edges(molecule, link, rev_raw_match):
    for from_node, to_node_attrs in link.non_edges:
        if from_node not in link:
            continue
        from_mol_node_name = rev_raw_match[from_node]
        from_mol = molecule.nodes[from_mol_node_name]
        from_link = link.nodes[from_node]
        from_resid = from_mol['resid']
        from_order = from_link.get('order', 0)
        for neighbor in molecule.neighbors(from_mol_node_name):
            to_mol = molecule.nodes[neighbor]
            to_link = to_node_attrs
            to_resid = to_mol['resid']
            to_order = to_link.get('order', 0)
            if to_resid == from_resid + to_order and _atoms_match(to_mol, to_link):
                return False
    return True


def _pattern_match(molecule, atoms, raw_match):
    for link_key, template_attr in atoms:
        molecule_key = raw_match[link_key]
        molecule_attr = molecule.nodes[molecule_key]
        if not _atoms_match(molecule_attr, template_attr):
            return False
    return True


def _any_pattern_match(molecule, patterns, rev_raw_match):
    return any(_pattern_match(molecule, atoms, rev_raw_match) for atoms in patterns)

        
def match_link(molecule, link):
    if not attributes_match(molecule.meta, link.molecule_meta):
        return

    GM = LinkGraphMatcher(molecule, link)

    raw_matches = GM.subgraph_isomorphisms_iter()
    for raw_match in raw_matches:
        # raw_match: mol -> link
        # rev_raw_match: link -> mol
        rev_raw_match = {value: key for key, value in raw_match.items()}
        if not _is_valid_non_edges(molecule, link, rev_raw_match):
            continue
        any_pattern_match = _any_pattern_match(molecule, link.patterns, rev_raw_match)
        if link.patterns and (not any_pattern_match):
            continue
        order_match = {}
        for mol_idx, link_idx in raw_match.items():
            mol_node = molecule.nodes[mol_idx]
            link_node = link.nodes[link_idx]
            if 'order' in link_node:
                order = link_node['order']
                resid = mol_node['resid']
                if order not in order_match:
                    order_match[order] = resid
                # Assert all orders correspond to the same resid
                elif order in order_match and order_match[order] != resid:
                    break
        else:  # No break
            for ((order1, resid1), (order2, resid2)) in combinations(order_match.items(), 2):
                # Assert the differences between orders correspond to
                # differences in resid
                if order2 - order1 != resid2 - resid1:
                    break
            else:  # No break
                # raw_match is molecule -> link. The other way around is more
                # usefull
                yield {v: k for k, v in raw_match.items()}


class DoLinks(Processor):
    def run_molecule(self, molecule):
        links = molecule.force_field.links
        for link in links:
            matches = match_link(molecule, link)
            for match in matches:
                for node, node_attrs in link.nodes.items():
                    if 'replace' in node_attrs:
                        node_mol = molecule.nodes[match[node]]
                        node_mol.update(node_attrs['replace'])
                for inter_type, params in link.removed_interactions.items():
                    for param in params:
                        param = param._replace(atoms=tuple(match[idx] for idx in param.atoms))
                        try:
                            molecule.remove_matching_interaction(inter_type, param)
                        except ValueError:
                            pass
                for inter_type, params in link.interactions.items():
                    for param in params:
                        param = param._replace(atoms=tuple(match[idx] for idx in param.atoms))
                        molecule.add_or_replace_interaction(inter_type, *param)
        return molecule
