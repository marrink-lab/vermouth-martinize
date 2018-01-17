"""
High level API for Martinize2
"""

import argparse
from pathlib import Path
import os
import numpy as np
import martinize2 as m2
from martinize2.forcefield import find_force_fields, FORCE_FIELDS
from martinize2 import DATA_PATH


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


def read_system(path):
    system = m2.System()
    file_extension = path.suffix.upper()[1:]  # We do not keep the dot
    if file_extension in ['PDB', 'ENT']:
        m2.PDBInput().run_system(system, str(path))
    elif file_extension in ['GRO']:
        m2.GROInput().run_system(system, str(path))
    else:
        raise ValueError('Unknown file extension "{}".'.format(file_extension))
    return system


def select_all(node):
    return True


def select_backbone(node):
    if node.get('atomname') == 'BB':
        return True
    return False


def pdb_to_universal(system):
    """
    Convert a system read from the PDB to a clean canonical atomistic system.
    """
    canonicalized = system.copy()
    canonicalized.force_field = FORCE_FIELDS['universal']
    m2.MakeBonds().run_system(canonicalized)
    m2.RepairGraph().run_system(canonicalized)
    return canonicalized


def martinize(system, mappings, to_ff):
    m2.DoMapping(mappings=mappings, to_ff=to_ff).run_system(system)
    m2.DoAverageBead().run_system(system)
    m2.ApplyBlocks().run_system(system)
    m2.DoLinks().run_system(system)
    return system


def entry():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='inpath', required=True, type=Path)
    parser.add_argument('-x', dest='outpath', required=True, type=Path)
    parser.add_argument('-p', dest='posres',
                        choices=('None', 'All', 'Backbone'), default='None')
    parser.add_argument('-pf', dest='posres_fc', type=float, default=500)
    parser.add_argument('-ff', dest='to_ff', default='martini22')
    args = parser.parse_args()

    known_force_fields = m2.forcefield.find_force_fields(Path(DATA_PATH) / 'force_fields')
    known_mappings = read_mapping_directory(Path(DATA_PATH) / 'mappings')

    from_ff = 'universal'
    if args.to_ff not in known_force_fields:
        raise ValueError('Unknown force field "{}".'.format(args.to_ff))
    if from_ff not in known_mappings or args.to_ff not in known_mappings[from_ff]:
        raise ValueError('No mapping known to go from "{}" to "{}".'
                         .format(from_ff, args.to_ff))

    # Reading the input structure.
    # So far, we assume we only go from atomistic to martini. We want the
    # input structure to be a clean universal system.
    system = read_system(args.inpath)
    system = pdb_to_universal(system)

    # Run martinize on the system.
    system = martinize(
        system,
        mappings=known_mappings,
        to_ff=known_force_fields[args.to_ff],
    )

    # Apply position restraints if required.
    if args.posres != 'None':
        selector = {'All': select_all, 'Backbone': select_backbone}[args.posres]
        m2.ApplyPosres(selector, args.posres_fc).run_system(system)

    # Write a PDB file.
    m2.pdb.write_pdb(system, str(args.outpath))

    for idx, molecule in enumerate(system.molecules):
        with open('molecule_{}.itp'.format(idx), 'w') as outfile:
            molecule.moltype = 'molecule_{}'.format(idx)
            molecule.nrexcl = 2
            m2.gmx.itp.write_molecule_itp(molecule, outfile)


if __name__ == '__main__':
    entry()
