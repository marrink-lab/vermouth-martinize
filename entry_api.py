"""
High level API for Martinize2
"""

import argparse
from pathlib import Path
import os
import numpy as np
import martinize2 as m2
from martinize2.forcefield import find_force_fields


DATA_DIR = Path(m2.__file__).parent / 'data'


class NullPositions(m2.processor.Processor):
    def run_molecule(self, molecule):
        '''
        Place atoms without position to the origin
        '''
        for node in molecule.nodes.values():
            if 'position' not in node:
                node['position'] = np.zeros((3,))
        return molecule


def read_mapping(path):
    from_ff = ['universal', ]
    to_ff = ['martini22', ]
    mapping = {}
    
    with open(str(path)) as infile:
        for line in infile:
            cleaned = line.split(';', 1)[0].strip()
            if not cleaned:
                continue
            elif cleaned[0] == '[':
                context = cleaned[1:-1].strip()
            elif context == 'molecule':
                name = cleaned
            elif context == 'atoms':
                _, from_atom, *to_atoms = line.split()
                mapping[(0, from_atom)] = [(0, to_atom) for to_atom in to_atoms]

    return name, from_ff, to_ff, mapping


def read_mapping_directory(directory):
    directory = Path(directory)
    mappings = {}
    for path in directory.glob('**/*.map'):
        name, all_from_ff, all_to_ff, mapping = read_mapping(path)
        for from_ff in all_from_ff:
            mappings[from_ff] = mappings.get(from_ff, {})
            for to_ff in all_to_ff:
                mappings[from_ff][to_ff] = mappings[from_ff].get(to_ff, {})
                mappings[from_ff][to_ff][name] = mapping
    return mappings


def read_system(infile):
    raise NotImplementedError()


def martinize(system, mappings, to_ff):
    # We start by only supporting the base usecase: going from an atomistic
    # system to a martini one. The input system is read from a PDB file or
    # from a GRO file, it is assumed not to have bonds; it is also assumed not
    # to have holes for the moment.

    # Since we do not have bonds, we need to guess them.
    m2.MakeBonds().run_system(system)

    # While we assume there is no major wholes in the structure, we can still
    # fix a few small things. Especially, we need to canonicalize the names and
    # maybe add the hydrogens.
    m2.RepairGraph().run_system(system)

    # At that point, we have a clean structure, so we can do the mapping.
    # The only available mapping now is the 1:1 mapping. We'll roll with this.
    m2.DoMapping(mappings=mappings, to_ff=to_ff).run_system(system)
    system.force_field = to_ff

    m2.ApplyBlocks().run_system(system)
    m2.DoLinks().run_system(system)

    return system


def entry():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='inpath', required=True)
    parser.add_argument('-x', dest='outpath', required=True)
    args = parser.parse_args()

    known_force_fields = m2.forcefield.find_force_fields(DATA_DIR / 'force_field')
    known_mappings = read_mapping_directory(DATA_DIR / 'mappings')

    system = m2.System()

    # Right now I only handle reading PDB files.
    m2.PDBInput().run_system(system, args.inpath)

    # I assume that I am reading an atomistic structure from the PDB.
    system.force_field = known_force_fields['universal']

    # Run martinize on the system.
    system = martinize(
        system,
        mappings=known_mappings,
        to_ff=known_force_fields['martini22'],
    )

    NullPositions().run_system(system)

    # Write a PDB file.
    m2.pdb.write_pdb(system, args.outpath)


if __name__ == '__main__':
    entry()
