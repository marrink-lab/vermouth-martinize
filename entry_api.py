# -*- coding: utf-8 -*-
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
    """
    Partial reader for Backward mapping files.

    ..warning::

        This parser is a limited proof of concept. It must be replaced! See
        [issue #5](https://github.com/jbarnoud/martinize2/issues/5).

    Read mapping from a Backward mapping file. Not all fields are supported,
    only the "molecule" and the "atoms" fields are read. The origin force field
    is assumed to be "universal", and the destination force field is assumed to
    be "martini22".

    There are no weight computed in case of shared atoms.

    The reader assumes only one molecule per file.

    Parameters
    ----------
    path: str or Path
        Path to the mapping file to read.

    Returns
    -------
    name: str
        The name of the fragment as read in the "molecule" field.
    from_ff: list of str
        A list of force field origins. Each force field is referred by name.
    to_ff: list of str
        A list of force field destinations. Each force field is referred by name.
    mapping: dict
        The mapping. The keys of the dictionary are the atom names in
        the origin force field; the values are lists of atom names in the
        destination force field, the origin atom is assigned to.
    """
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
    """
    Read all the mapping files in a directory.

    The resulting mapping collection is a 3-level dict where the keys are:
    * the name of the origin force field
    * the name of the destination force field
    * the name of the residue

    The values after these 3 levels is a mapping dict where the keys are the
    atom names in the origin force field and the values are lists of names in
    the destination force field.

    Parameters
    ----------
    directory: str or Path
        The path to the directory to search. Files with a '.map' extension will
        be read. There is no recursive search.

    Returns
    -------
    dict
        A collection of mappings.
    """
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
    """
    Read a system from a PDB or GRO file.

    This function guesses the file type based on the file extension.

    The resulting system does not have a force field and may not have edges.
    """
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
    """
    Convert a system from one force field to an other at lower resolution.
    """
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
