from os import system

import vermouth
from vermouth.processors.processor import Processor
import vermouth.pdb
import logging
from vermouth.log_helpers import TypeAdapter
from pathlib import Path
from vermouth import DATA_PATH
from vermouth.map_input import read_mapping_directory

LOGGER = TypeAdapter(logging.getLogger("vermouth"))
# import the martinize2 classes en functions 



def read_system(system, path, ignore_resnames=(), ignh=None, modelidx=None):
    """
    Read a system from a PDB or GRO file.

    This function guesses the file type based on the file extension.

    The resulting system does not have a force field and may not have edges.
    """
    file_extension = path.suffix.upper()[1:]  # We do not keep the dot
    if file_extension in ["PDB", "ENT"]:
        vermouth.PDBInput(
            str(path), exclude=ignore_resnames, ignh=ignh, modelidx=modelidx
        ).run_system(system)
    elif file_extension in ["GRO"]:
        vermouth.GROInput(str(path), exclude=ignore_resnames, ignh=ignh).run_system(
            system
        )
    elif file_extension in ["CIF"]:
        vermouth.CIFInput(str(path), exclude=ignore_resnames, ignh=ignh,
                          modelidx=modelidx).run_system(system)
    else:
        raise ValueError('Unknown file extension "{}".'.format(file_extension))
    return system



# define the processor class. readsystem itself does not have a forcefield. 
class ReadSystem(Processor):
    # constructor with parameters 
    def __init__(self, path, ignore_resnames=(), ignh=None, modelidx=None):
        # save the parameters 
        self.path = path
        self.ignore_resnames = ignore_resnames
        self.ignh = ignh
        self.modelidx = modelidx
    # define run system 
    def run_system(self, system):
        print("Running ReadSystem processor")
        print("Input file:", self.path)
        # fill the system with the parameters and return it
        system = read_system(
            system=system,
            path=self.path,
            ignore_resnames=self.ignore_resnames,
            ignh=self.ignh,
            modelidx=self.modelidx,
        )
        # print and return the system
        print("Returned system:", system)
        print(f'{system.force_field=}')
        return system

# write the current system to a pdb file. This is for testing and bug fixing, not a pipeline step. 
class WritePDB(Processor):
    # constructor with parameters
    def __init__(self, path=None, omit_charges = True, defer_writing = False, nan_missing_pos = True):
        # path to the output file 
        self.path = path
        # do you count charges 
        self.omit_charges = omit_charges
        # do you want to write it now or wait
        self.defer_writing = defer_writing
        # give nan to missing atom positions 
        self.nan_missing_pos = nan_missing_pos
    # run the system 
    def run_system(self, system):
        # if there is no path, do nothing. 
        if self.path is not None:
            # use the pdb writer of the vermouth library 
            vermouth.pdb.write_pdb(
                # the system to write
                system,
                # the path to write to
                str(self.path),
                # do you count charges, true means you dont 
                omit_charges=self.omit_charges,
                # do you want to write it now or wait
                defer_writing=self.defer_writing,
                # give nan to missing atom positions
                nan_missing_pos=self.nan_missing_pos,
            )
        # return a system, otherwise the pipeline will break.
        return system 

# this wrapper is made to combine multiple CLI into modifications and mutations 
class AnnotateMutModWrapper(Processor):
    # make the constructor with the parameters
    def __init__(self, modify=None, cter=None, nter=None, mutate=None):
        self.modify = modify or [] # use an empty list if modify is None
        self.cter = cter or []
        self.nter = nter or []
        self.mutate = mutate or []

    # call the system with the parameters
    def run_system(self, system):
        # make the list which will contain all the differnet modifications 
        modifications = []
        # loop through the modify arguments and split them into two on the :, add them to the modifications list. 
        for item in self.modify:
            modifications.append(item.split(':'))

        for item in self.cter:
            modifications.append(['cter', item])

        for item in self.nter:
            modifications.append(['nter', item])

        # mutations list 
        mutations = []
        for items in self.mutate: 
            mutations.append(items.split(':'))

        # check if modifications is empty
        if modifications:
            # split all modifications into two lists. 
            # resspecs is where needs to be and mods is which ones. 
            resspecs, mods = zip(*modifications)
        else:
            # if there are no modifications, make empty lists for resspecs and mods.
            resspecs, mods = [], []

        # If there are no modifications add the cter and nter so that the lists are not empty
        if not any("cter" in str(resspec) for resspec in resspecs):
            modifications.append(["cter", "+C-ter"])

        if not any("nter" in str(resspec) for resspec in resspecs):
            modifications.append(["nter", "+N-ter"])

        # call the processor with the modifications and mutations.
        vermouth.AnnotateMutMod(modifications, mutations).run_system(system)
        return system

class DoMappingWrapper(Processor):
    # constructor with parameters
    def __init__(self, to_ff, delete_unknown=False, attribute_keep=(), attribute_must=()):
        self.to_ff = to_ff
        self.delete_unknown = delete_unknown
        self.attribute_keep = attribute_keep
        self.attribute_must = attribute_must

    def run_system(self, system):
        # find the known forcefield and use that forcefield for mapping. 
        known_force_fields = vermouth.forcefield.find_force_fields(
            Path(DATA_PATH) / "force_fields"
        )
        # find the known mapping to use with the right forcefield. 
        known_mappings = read_mapping_directory(
            Path(DATA_PATH) / "mappings",
            known_force_fields
        )
        # say which forcefield to use 
        target_ff = known_force_fields[self.to_ff]
        vermouth.DoMapping(
            mappings = known_mappings,
            to_ff = target_ff,
            delete_unknown=self.delete_unknown,
            attribute_keep=self.attribute_keep,
            attribute_must=self.attribute_must
        ).run_system(system)
        return system

class Header(Processor):
    def __init__(self, version):
        self.version = version