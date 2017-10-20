# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:33:07 2017

@author: Peter Kroon
"""
from ..molecule import Molecule
from ..utils import first_alpha, distance
from ..truncating_formatter import TruncFormatter

import numpy as np


def write_pdb(graph, file_name, conect=True):
    def keyfunc(node_idx):
        return graph.node[node_idx]['chain'], graph.node[node_idx]['resid'], graph.node[node_idx]['resname']

    formatter = TruncFormatter()
#    format_string = 'ATOM  {: >5.5d} {:4.4s}{:1.1s}{:3.3s} {:1.1s}{:4.4d}{:1.1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:2.2s}{:2.2s}'
    format_string = 'ATOM  {: >5dt} {:4st}{:1st}{:3st} {:1st}{:>4dt}{:1st}   {:8.3ft}{:8.3ft}{:8.3ft}{:6.2ft}{:6.2ft}          {:2st}{:2st}'
    node_order = sorted(graph, key=keyfunc)
    nodeidx2atomid = {}
    with open(file_name, 'w') as out:
        for atomid, node_idx in enumerate(node_order, 1):
            nodeidx2atomid[node_idx] = atomid
            node = graph.node[node_idx]
            atomname = node['atomname']
            altloc = node.get('altloc', '')
            resname = node['resname']
            chain = node['chain']
            resid = node['resid']
            insertion_code = node.get('insertioncode', '')
            x, y, z = node['position']
            occupancy = node.get('occupancy', 1)
            temp_factor = node.get('temp_factor', 0)
            element = node.get('element', first_alpha(atomname))
            charge = '{:+2d}'.format(node.get('charge', 0))[::-1]
            line = formatter.format(format_string, atomid, atomname, altloc,
                                    resname, chain, resid, insertion_code, x,
                                    y, z, occupancy, temp_factor, element,
                                    charge)
            out.write(line + '\n')
        if conect:
            number_fmt = '{:>4dt}'
            format_string = 'CONECT '
            
            for node_idx in node_order:
                todo = [nodeidx2atomid[n_idx] for n_idx in graph[node_idx]]
                while todo:
                    current, todo = todo[:4], todo[4:]
                    fmt = ['CONECT'] + [number_fmt]*(len(current) + 1)
                    fmt = ' '.join(fmt)
                    line = formatter.format(fmt, nodeidx2atomid[node_idx], *current)
                    out.write(line + '\n')


def read_pdb(file_name, exclude=('SOL',), ignh=False):
    molecule = Molecule()
    idx = 0
    
    field_widths = (-6, 5, -1, 4, 1, 3, -1, 1, 4, 1, -3, 8, 8, 8, 6, 6, -10, 2, 2)
    field_types = (int, str, str, str, str, int, str, float, float, float, float, float, str, str)
    field_names = ('atomid', 'atomname', 'altloc', 'resname', 'chain', 'resid',
                   'insertion_code', 'x', 'y', 'z', 'occupancy', 'temp_factor',
                   'element', 'charge')

    start = 0
    slices = []
    for width in field_widths:
        if width > 0:
            slices.append(slice(start, start + width))
        start = start + abs(width)
    
    with open(file_name) as pdb:
        for line in pdb:
            record = line[:6]
            if record == 'ATOM  ' or record == 'HETATM':
                properties = {}
                for name, type_, slice_ in zip(field_names, field_types, slices):
                    properties[name] = type_(line[slice_].strip())

                properties['position'] = np.array((properties['x'], properties['y'], properties['z']), dtype=float)
                del properties['x']
                del properties['y']
                del properties['z']
                if not properties['element']:
                    atomname = properties['atomname']
                    properties['element'] = first_alpha(atomname)
                if properties['resname'] in exclude or (ignh and properties['element'] == 'H'):
                    continue
                molecule.add_node(idx, **properties)
                idx += 1
            elif record == 'CONECT':
                atidx2nodeidx = {node_data['atomid']: node_idx
                                 for node_idx, node_data in molecule.node.items()}
                start = 6
                width = 5
                ats = []
                for num in range(5):
                    try:
                        at = int(line[start + num*width:start + (num + 1)*width])
                        ats.append(at)
                    except (IndexError, ValueError):
                        # We ran out of line or read a bit of whitespace
                        pass

                try:
                    at0 = atidx2nodeidx[ats[0]]
                    for at in ats[1:]:
                        at = atidx2nodeidx[at]

                        w = 1/distance(molecule.node[at0]['position'], molecule.node[at]['position'])
                        molecule.add_edge(at0, at, weight=w)
                except KeyError:
                    pass

#    if molecule.number_of_edges() == 0:
#        edges_from_distance(molecule)
#        # Make all edges based on threshold distance
#        positions = np.array([molecule.node[n]['position'] for n in molecule])
#        distances = ssd.squareform(ssd.pdist([molecule.node[n]['position'] for n in molecule]))
#        idxs = np.where((distances < threshold) & (distances != 0))
#        weights = 1/distances[idxs]
#        molecule.add_weighted_edges_from(zip(idxs[0], idxs[1], weights))
    return molecule