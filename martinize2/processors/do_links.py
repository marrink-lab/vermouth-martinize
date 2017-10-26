#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:00:02 2017

@author: peterkroon
"""
from .processor import Processor
from ..gmx import read_rtp

from itertools import combinations

import networkx as nx


class LinkGraphMatcher(nx.isomorphism.GraphMatcher):
    def semantic_feasibility(self, node1, node2):
        # TODO: implement (partial) wildcards
        # Node2 is the link
        node1 = self.G1.nodes[node1]
        node2 = self.G2.nodes[node2]
        for attr in node2:
            if attr == 'order':
                continue
            if node1.get(attr, None) != node2[attr]:
                return False
        else:
            return True


def match_link(molecule, link):
    GM = LinkGraphMatcher(molecule, link)

    raw_matches = GM.subgraph_isomorphisms_iter()
    for raw_match in raw_matches:
        # mol -> link
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


RTP_PATH = '/usr/local/gromacs-2016.3/share/gromacs/top/charmm27.ff/aminoacids.rtp'


class DoLinks(Processor):
    def run_molecule(self, molecule):
        with open(RTP_PATH) as rtp:
            blocks, links = read_rtp(rtp)
        for link in links:
            matches = match_link(molecule, link)
            for match in matches:
                for inter_type, params in link.interactions:
                    params = params._replace(atoms=tuple(match[idx] for idx in params.atoms))
                    molecule.add_interaction(inter_type, *params)
        return molecule
