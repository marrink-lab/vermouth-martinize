# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:34:12 2017

@author: Peter Kroon
"""

def read_gro(file_name, exclude=('SOL',), ignh=False):
    molecule = nx.Graph()
    idx = 0
    field_widths = (5, 5, 5, 5, 8, 8, 8, 8, 8, 8)
    field_types = (int, str, str, int, float, float, float, float, float, float)
    field_names = ('resid', 'resname', 'atomname', 'atomid', 'x', 'y', 'z', 'vx', 'vy', 'vz')

    start = 0
    slices = []
    for width in field_widths:
        if width > 0:
            slices.append(slice(start, start + width))
        start = start + abs(width)

    with open(file_name) as gro:
        next(gro)  # useless header line
        num_atoms = int(next(gro))  # Not sure we'll use this
        for line_idx, line in enumerate(gro):
            properties = {}
            try:
                for name, type_, slice_ in zip(field_names, field_types, slices):
                    properties[name] = type_(line[slice_].strip())
            except ValueError:  # box specifications.
                if line_idx != num_atoms:
                    print(len(molecule), num_atoms)
                    print(line)
                    raise
                continue
            properties['position'] = np.array((properties['x'], properties['y'], properties['z']), dtype=float)
            properties['position'] *= 10  # Convert nm to A
            del properties['x']
            del properties['y']
            del properties['z']
            properties['element'] = first_alpha(properties['atomname'])
            properties['chain'] = ''
            if properties['resname'] in exclude or (ignh and properties['element'] == 'H'):
                continue

            molecule.add_node(idx, **properties)
            idx += 1
    assert line_idx == num_atoms
    edges_from_distance(molecule)
#    positions = np.array([molecule.node[n]['position'] for n in molecule])
#    # This does the same as scipy.spatial.distance.squareform(pdist(positions))
#    distances = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], ord=2, axis=2)
#    idxs = np.where((distances < threshold) & (distances != 0))
#    weights = 1/distances[idxs]
#    molecule.add_weighted_edges_from(zip(idxs[0], idxs[1], weights))

    return molecule