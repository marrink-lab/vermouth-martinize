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
High level API for Martinize2
"""

import functools
import logging
import itertools
import textwrap
from pathlib import Path
import sys

from . import __version__
from .system import System
from .molecule import Molecule
from .pdb import write_pdb
from .gmx.itp import write_molecule_itp
from . import processors
from . import forcefield
from .file_writer import open, DeferredFileWriter
from . import DATA_PATH
from .dssp import dssp
from .dssp.dssp import (
    AnnotateDSSP,
    AnnotateMartiniSecondaryStructures,
    AnnotateResidues,
)
from .log_helpers import (StyleAdapter, BipolarFormatter,
                          CountingHandler, TypeAdapter,
                          ignore_warnings_and_count,)
from . import selectors
from .map_input import (
    read_mapping_directory,
    generate_all_self_mappings,
    combine_mappings
)
from .citation_parser import citation_formatter

# TODO Since vermouth's __init__.py does some logging (KDTree), this may or may
# not work as intended. Investigation required.

LOGGER = TypeAdapter(logging.getLogger('vermouth'))

PRETTY_FORMATTER = logging.Formatter(fmt='{levelname:>8} - {type} - {message}',
                                     style='{')
DETAILED_FORMATTER = logging.Formatter(fmt='{levelname:>8} - {type} - {name} - {message}',
                                       style='{')

COUNTER = CountingHandler()

# Control above what level message we want to count
COUNTER.setLevel(logging.WARNING)

CONSOLE_HANDLER = logging.StreamHandler()
FORMATTER = BipolarFormatter(DETAILED_FORMATTER,
                             PRETTY_FORMATTER,
                             logging.DEBUG,
                             logger=LOGGER)
CONSOLE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(CONSOLE_HANDLER)
LOGGER.addHandler(COUNTER)

LOGGER = StyleAdapter(LOGGER)

VERSION = 'martinize with vermouth {}'.format(__version__)


def read_system(path, ignore_resnames=(), ignh=None, modelidx=None):
    """
    Read a system from a PDB or GRO file.

    This function guesses the file type based on the file extension.

    The resulting system does not have a force field and may not have edges.
    """
    system = System()
    file_extension = path.suffix.upper()[1:]  # We do not keep the dot
    if file_extension in ['PDB', 'ENT']:
        processors.PDBInput(str(path), exclude=ignore_resnames, ignh=ignh,
                          modelidx=modelidx).run_system(system)
    elif file_extension in ['GRO']:
        processors.GROInput(str(path), exclude=ignore_resnames, ignh=ignh).run_system(system)
    else:
        raise ValueError('Unknown file extension "{}".'.format(file_extension))
    return system


def pdb_to_universal(system, delete_unknown=False, force_field=None,
                     modifications=None, mutations=None,
                     bonds_from_name=True, bonds_from_dist=True, bonds_fudge=1,
                     write_graph=None, write_repair=None, write_canon=None):
    """
    Convert a system read from the PDB to a clean canonical atomistic system.
    """
    if force_field is None:
        force_field = forcefield.get_native_force_field('universal')
    if modifications is None:
        modifications = []
    if mutations is None:
        mutations = []
    canonicalized = system.copy()
    canonicalized.force_field = force_field

    LOGGER.info('Guessing the bonds.', type='step')
    processors.MakeBonds(allow_name=bonds_from_name,
                       allow_dist=bonds_from_dist,
                       fudge=bonds_fudge).run_system(canonicalized)
    processors.MergeNucleicStrands().run_system(canonicalized)
    if write_graph is not None:
        write_pdb(canonicalized, str(write_graph), omit_charges=True)
        DeferredFileWriter().write()

    LOGGER.debug('Annotating required mutations and modifications.', type='step')
    processors.AnnotateMutMod(modifications, mutations).run_system(canonicalized)
    LOGGER.info('Repairing the graph.', type='step')
    processors.RepairGraph(delete_unknown=delete_unknown, include_graph=False).run_system(canonicalized)
    if write_repair is not None:
        write_pdb(canonicalized, str(write_repair),
                               omit_charges=True, nan_missing_pos=True)
        DeferredFileWriter().write()
    LOGGER.info('Dealing with modifications.', type='step')
    processors.CanonicalizeModifications().run_system(canonicalized)
    if write_canon is not None:
        write_pdb(canonicalized, str(write_canon),
                               omit_charges=True, nan_missing_pos=True)
        DeferredFileWriter().write()
    processors.AttachMass(attribute='mass').run_system(canonicalized)
    processors.SortMoleculeAtoms().run_system(canonicalized) # was system
    return canonicalized


def martinize(system, mappings, to_ff, delete_unknown=False):
    """
    Convert a system from one force field to an other at lower resolution.
    """
    LOGGER.info('Creating the graph at the target resolution.', type='step')
    processors.DoMapping(mappings=mappings,
                       to_ff=to_ff,
                       delete_unknown=delete_unknown,
                       attribute_keep=('cgsecstruct', 'resid', 'chain'),
                       attribute_must=('resname')).run_system(system)
    LOGGER.info('Averaging the coordinates.', type='step')
    processors.DoAverageBead(ignore_missing_graphs=True).run_system(system)
    LOGGER.info('Applying the links.', type='step')
    processors.DoLinks().run_system(system)
    LOGGER.info('Placing the charge dummies.', type='step')
    processors.LocateChargeDummies().run_system(system)
    return system


def write_gmx_topology(system, top_path, defines=(), header=()):
    """
    Writes a Gromacs .top file for the specified system.
    """
    if not system.molecules:
        raise ValueError('No molecule in the system. Nothing to write.')

    # Write the ITP files for the molecule types, and prepare writing the
    # [ molecules ] section of the top file.
    # * We write one ITP file for each different moltype in the system, the
    #   moltype being defined by the name provided under the "moltype" meta of
    #   the molecules. If more than one molecule share the same moltype, we use
    #   the first one to write the ITP file.
    moltype_written = set()
    # * We keep track of the length of the longer moltype name, to align the
    #   [ molecules ] section.
    max_name_length = 0
    # * We keep track of groups of successive molecules with the same moltypes.
    moltype_count = []  # items will be [moltype, number of molecules]

    # Iterate over groups of successive molecules with the same moltypes. We
    # shall *NOT* sort the molecules before hand, as groups of successive
    # molecules with the same moltype can be interupted by other moltypes, and
    # we want to reflect these interuptions in the [ molecules ] section of the
    # top file.
    molecule_groups = itertools.groupby(system.molecules, key=lambda x: x.meta['moltype'])
    for moltype, molecules in molecule_groups:
        molecule = next(molecules)
        if moltype not in moltype_written:
            # A given moltype can appear more than once in the sequence of
            # molecules, without being uninterupted by other moltypes. Even in
            # that case, we want to write the ITP only once.
            with open('{}.itp'.format(moltype), 'w') as outfile:
                # here we format and merge all citations
                header[-1] = header[-1]+"\n"
                header.append("Pleas cite the following papers:")
                for citation in molecule.citations:
                    cite_string =  citation_formatter(molecule.force_field.citations[citation])
                    LOGGER.info("Please cite: " + cite_string)
                    header.append(cite_string)
                write_molecule_itp(molecule, outfile, header=header)
            this_moltype_len = len(molecule.meta['moltype'])
            if this_moltype_len > max_name_length:
                max_name_length = this_moltype_len
            moltype_written.add(moltype)
        # We already removed one element from the "molecules" generator, do not
        # forget to count it in the number of molecules in that group.
        moltype_count.append([moltype, 1 + len(list(molecules))])

    # Write the top file
    template = textwrap.dedent("""\
        {defines}
        #include "martini.itp"
        {includes}

        [ system ]
        Title of the system

        [ molecules ]
        {molecules}
    """)
    include_string = '\n'.join(
        '#include "{}.itp"'.format(molecule_type)
        for molecule_type, _ in moltype_count
    )
    molecule_string = '\n'.join(
        '{mtype:<{length}}    {num}'
        .format(mtype=mtype, num=num, length=max_name_length)
        for mtype, num in moltype_count
    )
    define_string = '\n'.join(
        '#define {}'.format(define) for define in defines
    )
    with open(str(top_path), 'w') as outfile:
        outfile.write(
            textwrap.dedent(
                template.format(
                    includes=include_string,
                    molecules=molecule_string,
                    defines=define_string,
                )
            )
        )




def martinize2(
    inpath, outpath, top_path=None, keep_duplicate_itp=False,
    merge_chains=[], ignore_res=[], ignore_h=False, modelidx=None,
    bonds_from='both', bonds_fudge=1.2, to_ff='martini3001', from_ff='universal',
    extra_ff_dir=[], extra_map_dir=[], list_ff=False, list_blocks=False,
    posres='none', posres_fc=1000., dssp_exe=None, ss=None, collagen=False,
    extdih=False, elastic=False, rb_force_constant=500., rb_lower_bound=0.,
    rb_upper_bound=0.9, res_min_dist=None, rb_decay_factor=0., rb_decay_power=1.,
    rb_minimum_force=0., rb_selection=None, rb_unit='molecule', govs_includes=False,
    govs_moltype='molecule_0', scfix=False, cystein_bridge='none',
    mutations=[], modifications=[], neutral_termini=False, write_graph=None,
    write_repair=None, write_canon=None, verbosity=0, maxwarn=[]
    ):
    """
    High level API for Martinize2

    Parameters
    ----------
    inpath: str or libpath.Path
        Input file (PDB|GRO)
    outpath: str or libpath.Path
        Output coarse grained structure (PDB)
    top_path: str or libpath.Path
        Write separate topologies for identical chains
    keep_duplicate_itp: bool, default=False
        Write separate topologies for identical chains
    merge_chains: list of list of str, optional
        Merge specified chains.
        Ex: [['A','B'], ['C', 'D', 'E']]
    ignore_res: list of list of str, optional
        Ignore residues with that name.
        Ex: [['HOH','LIG']]
    ignore_h: bool, default=False
        Ignore all Hydrogen atoms in the input file
    modelidx: int, optional
        Which MODEL to select. Only meaningful for PDB files.
    bonds_from: {'both', 'name', 'distance', 'none'}, optional
        How to determine connectivity in the input.
        If 'none', only bonds from the input file (CONECT) will be used.
    bonds_fudge: float, default=1.2
        Factor with which Van der Waals radii should be scaled when
        determining bonds based on distances.
    
    Force field parameters
    ----------------------
    to_ff: str, default=martini3001
        Which forcefield to use.
        To know available forcefields:
        >>> forcefield.find_force_fields(
        ...     Path(vermouth.DATA_PATH) / 'force_fields')

    from_ff: str, default='universal'
        Force field of the original structure
    extra_ff_dir: list of str or Path, optional
        Additional repository for custom force fields.
    extra_map_dir: list of str or Path, optional
        Additional repository for mapping files.
    list_ff: bool, optional
        List all known force fields, and exit.
    list_blocks: bool, optional
        List all Blocks and Modifications known to the force field, and
        exit.

    Position restraints parameters
    ------------------------------
    posres: {'none', 'all', 'backbone'}
        Output position restraints.
    posres_fc: float, default=1000.
        Position restraints force constant in kJ/mol/nm^2.
    
    Secondary structure parameters
    ------------------------------
    dssp_exe: str, optional
        DSSP executable for determining structure.
    ss: str, optional
        Manually set the secondary structure of the proteins.
    collagen: bool, default=False
        Use collagen parameters
    extdih: bool, default=False

    Elastic network parameters
    --------------------------
    elastic: bool, default=False
        Write elastic bonds
    rb_force_constant: float, optional
        Elastic bond force constant Fc in kJ/mol/nm^2
    rb_lower_bound: float, optional
        Elastic bond lower cutoff: F = Fc if rij < lo
    rb_upper_bound: float, optional
        Elastic bond upper cutoff: F = 0  if rij > up
    res_min_dist: int, optional
        The minimum separation between two residues to have an RB
        the default value is set by the force-field
    rb_decay_factor: float, optional
        Elastic bond decay factor a.
    rb_decay_power
        Elastic bond decay power p.
    rb_minimum_force 
        Remove elastic bonds with force constant lower than this
    rb_selection
        Comma separated list of bead names for elastic bonds
    rb_unit: {'molecule', 'chain', 'all'}
        Establish what is the structural unit for the 'elastic network.
        Bonds are only created within a unit.

    GoMartini parameters
    --------------------
    govs_includes: bool, optional
    govs_moltype: str, optional
        Set the name of the molecule when using Virtual Sites GoMartini.

    Protein description parameters
    ------------------------------
    scfix: bool, optional
        Apply side chain corrections.
    cystein_bridge: {'none', 'auto', float}
        Disulfide bridges distance in nanometers. Default is 'none',
        'auto' will find the distance by itself, else specify the
        distance to detect cysteines.
    mutations: list of list of str, optional
        Mutate a residue. Ex: [['A-PHE45', 'ALA']].
        The format is <chain>-<resname><resid>:<new resname>. Elements
        of the specification can be omitted as required.
    modifications: list of list of str, optional
        Ex: [['A-ASP45','ASP0'], ['nter', 'NH3-ter'], ['cter', 'COOH-ter']]
        Add modifications to residues. Can also specify N termini and
        C termini types. 
        The format is <chain>-<resname><resid>:<modification>. Elements
        of the specification can be omitted as required.
        Values for nter: {'N-ter', 'NH2-ter', ...}
        Values for cter: {'C-ter', 'COOH-ter', ...}
        Set list_blocks to True for other values.
    neutral_termini: bool, optional
        Set neutral termini (charged is default).
        Alias for modifications=[['nter','NH2-ter'],['cter','COOH-ter']]
        Priority over modifications.
    
    Debugging parameters
    --------------------
    write_graph: str or Path, optional
        Write the graph as PDB after the MakeBonds step.
    write_repair: str or Path, optional
        Write the graph as PDB after the RepairGraph step. The resulting
        file may contain "nan" coordinates making it unreadable by most
        softwares. 
    write_canon: str or Path, optional
        Write the graph as PDB after the CanonicalizeModifications step.
        The resulting file may contain "nan" coordinates making it
        unreadable by most software.
    verbosity: int, default=0
        Enable debug logging output. Can be given multiple times.
    maxwarn: list of str, optional
        The maximum number of allowed warnings. If more warnings are
        encountered no output files are written.
    """
    if elastic and govs_includes:
        raise ValueError('A rubber band elastic network and GoMartini '
                         'are not compatible. The elastic and govs_include '
                         'arguments cannot be used together.')

    if to_ff.startswith('elnedyn'):
        # FIXME: This type of thing should be added to the FF itself.
        LOGGER.info('The forcefield {} must always be used with an elastic '
                    'network. Enabling it now.', to_ff)
        elastic = True

    file_extension = inpath.suffix.upper()[1:]  # We do not keep the dot
    if file_extension in ['GRO'] and modelidx is not None:
        raise ValueError("GRO files don't know the concept of models.")
    if modelidx is None:
        # Set a sane default value. Can't do this using argparse machinery,
        # since we need to be able to check whether the flag was given.
        modelidx = 1

    bonds_from_name = bonds_from in ('name', 'both')
    bonds_from_dist = bonds_from in ('distance', 'both')

    loglevels = {0: logging.INFO, 1: logging.DEBUG, 2: 5}
    LOGGER.setLevel(loglevels[verbosity])

    known_force_fields = forcefield.find_force_fields(
        Path(DATA_PATH) / 'force_fields'
    )
    known_mappings = read_mapping_directory(Path(DATA_PATH) / 'mappings',
                                            known_force_fields)

    # Add user force fields and mappings
    for directory in extra_ff_dir:
        try:
            forcefield.find_force_fields(directory, known_force_fields)
        except FileNotFoundError:
            msg = '"{}" given to the -ff-dir option should be a directory.'
            raise ValueError(msg.format(directory))
    for directory in extra_map_dir:
        try:
            partial_mapping = read_mapping_directory(directory,
                                                     known_force_fields)
        except NotADirectoryError:
            msg = '"{}" given to the -map-dir option should be a directory.'
            raise ValueError(msg.format(directory))
        combine_mappings(known_mappings, partial_mapping)

    if list_ff:
        print('The following force fields are known:')
        for idx, ff_name in enumerate(reversed(list(known_force_fields)), 1):
            print('{:3d}. {}'.format(idx, ff_name))
        sys.exit()

    # Build self mappings
    partial_mapping = generate_all_self_mappings(known_force_fields.values())
    combine_mappings(known_mappings, partial_mapping)

    if to_ff not in known_force_fields:
        raise ValueError('Unknown force field "{}".'.format(to_ff))
    if from_ff not in known_force_fields:
        raise ValueError('Unknown force field "{}".'.format(from_ff))
    #if from_ff not in known_mappings or to_ff not in known_mappings[from_ff]:
    #    raise ValueError('No mapping known to go from "{}" to "{}".'
    #                     .format(from_ff, to_ff))

    if list_blocks:
        print('The following Blocks are known to force field {}:'.format(from_ff))
        print(', '.join(known_force_fields[from_ff].blocks))
        print('The following Modifications are known to force field {}:'.format(from_ff))
        print(', '.join(known_force_fields[from_ff].modifications))
        print()
        print('The following Blocks are known to force field {}:'.format(to_ff))
        print(', '.join(known_force_fields[to_ff].blocks))
        print('The following Modifications are known to force field {}:'.format(to_ff))
        print(', '.join(known_force_fields[to_ff].modifications))
        sys.exit()


    # ignore_res is a pretty deep list: given "-ignore HOH CU,LIG -ignore LIG2"
    # it'll contain [[['HOH'], ['CU', 'LIG']], [['LIG2']]]
    ignore_res = set()
    for grp in ignore_res:
        ignore_res.update(*grp)

    if neutral_termini:
        modifications.append(['cter', 'COOH-ter'])
        modifications.append(['nter', 'NH2-ter'])
    else:
        if modifications:
            resspecs, mods = zip(*modifications)
        else:
            resspecs, mods = [], []
        if not any('cter' in resspec for resspec in resspecs):
            modifications.append(['cter', 'C-ter'])
        if not any('nter' in resspec for resspec in resspecs):
            modifications.append(['nter', 'N-ter'])

    # Reading the input structure.
    # So far, we assume we only go from atomistic to martini. We want the
    # input structure to be a clean universal system.
    # For now at least, we silently delete molecules with unknown blocks.
    system = read_system(inpath, ignore_resnames=ignore_res,
                         ignh=ignore_h, modelidx=modelidx)
    system = pdb_to_universal(
        system,
        delete_unknown=True,
        force_field=known_force_fields[from_ff],
        bonds_from_name=bonds_from_name,
        bonds_from_dist=bonds_from_dist,
        bonds_fudge=bonds_fudge,
        modifications=modifications,
        mutations=mutations,
        write_graph=write_graph,
        write_repair=write_repair,
        write_canon=write_canon,
    )

    LOGGER.info('Read input.', type='step')
    for molecule in system.molecules:
        LOGGER.debug("Read molecule {}.", molecule, type='step')

    target_ff = known_force_fields[to_ff]
    if dssp is not None:
        AnnotateDSSP(executable=dssp_exe, savedir='.').run_system(system)
        AnnotateMartiniSecondaryStructures().run_system(system)
    elif ss is not None:
        AnnotateResidues(attribute='secstruct', sequence=ss,
                         molecule_selector=selectors.is_protein).run_system(system)
        AnnotateMartiniSecondaryStructures().run_system(system)
    elif collagen:
        if not target_ff.has_feature('collagen'):
            LOGGER.warning('The force field "{}" does not have specific '
                           'parameters for collagen (-collagen).',
                           target_ff.name, type='missing-feature')
        AnnotateResidues(attribute='cgsecstruct', sequence='F',
                         molecule_selector=selectors.is_protein).run_system(system)
    if extdih and not target_ff.has_feature('extdih'):
        LOGGER.warning('The force field "{}" does not define dihedral '
                       'angles for extended regions of proteins (-extdih).',
                       target_ff.name, type='missing-feature')
    processors.SetMoleculeMeta(extdih=extdih).run_system(system)
    if scfix and not target_ff.has_feature('scfix'):
        LOGGER.warning('The force field "{}" does not define angle and '
                       'torsion for the side chain corrections (-scfix).',
                       target_ff.name, type='missing-feature')
    processors.SetMoleculeMeta(scfix=scfix).run_system(system)

    ss_sequence = list(itertools.chain(*(
        dssp.sequence_from_residues(molecule, 'secstruct')
        for molecule in system.molecules
        if selectors.is_protein(molecule)
    )))

    if cystein_bridge == 'none':
        processors.RemoveCysteinBridgeEdges().run_system(system)
    elif cystein_bridge != 'auto':
        processors.AddCysteinBridgesThreshold(cystein_bridge).run_system(system)

    # Run martinize on the system.
    system = martinize(
        system,
        mappings=known_mappings,
        to_ff=known_force_fields[to_ff],
        delete_unknown=True,
    )

    # Apply position restraints if required.
    if posres != 'none':
        LOGGER.info('Applying position restraints.', type='step')
        node_selectors = {'all': selectors.select_all,
                          'backbone': selectors.select_backbone}
        node_selector = node_selectors[posres]
        processors.ApplyPosres(node_selector, posres_fc).run_system(system)

    if govs_includes:
        # The way Virtual Site GoMartini works has to be in sync with
        # Sebastian's create_goVirt.py script, until the method is fully
        # implemented in vermouth. One call of martinize2 must create a single
        # molecule, regardless of the number of fragments in the input.
        # The molecule type name is provided as an input with the -govs-moltype
        # flag to be consistent with the name provided to Sebastian's script.
        # The name cannot be guessed because a system may need to be composed
        # from multiple calls to martinize2 and create_goVirt.py.
        LOGGER.info('Adding includes for Virtual Site Go Martini.', type='step')
        LOGGER.info('The output topology will require files generated by '
                    '"create_goVirt.py".')
        processors.MergeAllMolecules().run_system(system)
        processors.SetMoleculeMeta(moltype=govs_moltype).run_system(system)
        processors.GoVirtIncludes().run_system(system)
        defines = ('GO_VIRT',)
    else:
        # Merge chains if required.
        if merge_chains:
            for chain_set in merge_chains:
                processors.MergeChains(chain_set).run_system(system)
        processors.NameMolType(deduplicate=not keep_duplicate_itp).run_system(system)
        defines = ()

    # Apply a rubber band elastic network is required.
    if elastic:
        LOGGER.info('Setting the rubber bands.', type='step')
        if rb_unit == 'molecule':
            domain_criterion = processors.apply_rubber_band.always_true
        elif rb_unit == 'all':
            processors.MergeAllMolecules().run_system(system)
            domain_criterion = processors.apply_rubber_band.always_true
        elif rb_unit == 'chain':
            domain_criterion = processors.apply_rubber_band.same_chain
        else:
            message = 'Unknown value for -eunit: "{}".'.format(rb_unit)
            LOGGER.critical(message)
            raise ValueError(message)
        if rb_selection is not None:
            selector = functools.partial(
                selectors.proto_select_attribute_in,
                attribute='atomname',
                values=rb_selection,
            )
        else:
            selector = selectors.select_backbone
        rubber_band_processor = processors.ApplyRubberBand(
            lower_bound=rb_lower_bound,
            upper_bound=rb_upper_bound,
            decay_factor=rb_decay_factor,
            decay_power=rb_decay_power,
            base_constant=rb_force_constant,
            minimum_force=rb_minimum_force,
            selector=selector,
            domain_criterion=domain_criterion,
            res_min_dist=res_min_dist
        )
        rubber_band_processor.run_system(system)

    LOGGER.info('Writing output.', type='step')
    for molecule in system.molecules:
        LOGGER.debug("Writing molecule {}.", molecule, type='step')

    # Write the topology if requested
    # grompp has a limit in the number of character it can read per line
    # (due to the size limit of a buffer somewhere in its implementation).
    # The command line can be longer than this limit and therefore
    # prevent grompp from reading the topology.
    gromacs_char_limit = 4000  # the limit is actually 4095, but I play safe
    command = ' '.join(sys.argv)
    if len(command) > gromacs_char_limit:
        command = command[:gromacs_char_limit] + ' ...'
    header = [
        'This file was generated using the following command:',
        command,
        VERSION,
    ]
    if None not in ss_sequence:
        header += [
            'The following sequence of secondary structure ',
            'was used for the full system:',
            ''.join(ss_sequence),
        ]

    if top_path is not None:
        write_gmx_topology(system, top_path, defines=defines, header=header)

    # Write a PDB file.
    write_pdb(system, str(outpath), omit_charges=True)

    # TODO: allow ignoring warnings per class/amount (i.e. ignore 2
    #       inconsistent-data warnings)

    leftover_warnings = ignore_warnings_and_count(COUNTER, maxwarn)
    if leftover_warnings:
        LOGGER.error('{} warnings were encountered after accounting for the '
                     '-maxwarn flag. No output files will be '
                     'written. Consider fixing the warnings, or if you are sure'
                     ' they are harmless, use the -maxwarn flag.', leftover_warnings)
        sys.exit(2)
    else:
        DeferredFileWriter().write()
        processors.Quoter().run_system(system)