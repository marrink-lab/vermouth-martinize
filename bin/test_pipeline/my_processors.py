from os import system
import sys 
import vermouth
from vermouth.processors.processor import Processor
import vermouth.pdb
import logging
from vermouth.log_helpers import TypeAdapter
from pathlib import Path
from vermouth import DATA_PATH
from vermouth.map_input import read_mapping_directory
import vermouth
import functools
from vermouth import selectors
from vermouth.rcsu.contact_map import read_go_map, GenerateContactMap
from vermouth.rcsu.go_pipeline import GoPipeline

VERSION = "martinize with vermouth {}".format(vermouth.__version__)
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
    def run_system(self, system):
        # add a header to system with all command line arguments and the used version 
        system.meta["header"].extend((
            "This file was generated using the following command:",
            " ".join(sys.argv),
            VERSION,
        ))
        return system
    
# secundairy structure options. you can give only 1 out of 3. 
class DSSPWrapper(Processor):
    def __init__(self, executable = None, savedir = "."):
        self.executable = executable
        self.savedir = savedir
    def run_system(self, system):
        executable = self.executable
        # if you dont give a string, so nothing, it automatically uses mdtraj, otherwise you can give a path or program file. 
        if not isinstance(executable, str):
            executable = None 
        vermouth.dssp.dssp.AnnotateDSSP(executable = executable, savedir = self.savedir).run_system(system)
        vermouth.dssp.dssp.AnnotateMartiniSecondaryStructures().run_system(system)
        return system

class SSWrapper(Processor):
    def __init__(self, ss = None):
        self.ss = ss
    def run_system(self, system):
        if self.ss is None:
            return system 
        # convert everything the user gives in -ss to uppercase, otherwise it doesnt work apperently. 
        sequence = self.ss.upper()

        vermouth.dssp.dssp.AnnotateResidues(
            attribute = "aasecstruct",
            sequence = sequence,
            molecule_selector = selectors.is_protein,
        ).run_system(system)
        vermouth.dssp.dssp.AnnotateMartiniSecondaryStructures().run_system(system)
        return system 

class CollagenWrapper(Processor):
    def run_system(self, system):
        vermouth.dssp.dssp.AnnotateResidues(
            attribute="cgsecstruct",
            sequence="F",
            # wrapper was needed for this step, because you cant place that logic into yaml. 
            molecule_selector=selectors.is_protein,
        ).run_system(system)
        return system
    
class GoWrapper(Processor):
    def __init__(self, go, go_write_file = None):
        self.go = go
        self.go_write_file = go_write_file
    def run_system(self, system):
        if isinstance(self.go, Path):
            LOGGER.info("Reading Go model contact map.", type="step")
            read_go_map(system=system, file_path=self.go)
        else:
            LOGGER.info("Generating Go model contact map.", type="step")
            GenerateContactMap(write_file=self.go_write_file).run_system(system)
        return system
    
class RTPPolisherWrapper(Processor):
    def __init__(self, to_ff):
        self.to_ff = to_ff
    def run_system(self, system):
        known_force_fields = vermouth.forcefield.find_force_fields(
            Path(DATA_PATH) / "force_fields"
        )
        if 'bondedtypes' in known_force_fields[self.to_ff].variables:
            LOGGER.info("Generating implicit interactions for RTP force field", type='step')
            vermouth.RTPPolisher().run_system(system)
        return system
    
class ApplyPosresWrapper(Processor):
    def __init__(self, posres, posres_fc):
        self.posres = posres
        self.posres_fc = posres_fc
    def run_system(self, system):
        LOGGER.info("Applying position restraints.", type="step")
        node_selectors = {
            "all": (selectors.select_all, None),
            "backbone": (selectors.select_backbone, system.force_field.variables['bb_atomname'])
        }
        node_selector = node_selectors[self.posres]
        vermouth.ApplyPosres(node_selector, self.posres_fc).run_system(system)

class GoModelWrapper(Processor):
    def __init__ (
            self,
            go_low,
            go_up,
            go_eps,
            go_res_dist,
            go_backbone,
            go_atomname,
            molname,
            water_bias = False,
            water_bias_eps = None,
            water_bias_idrs = None
        ):
        self.go_low = go_low
        self.go_up = go_up
        self.go_eps = go_eps
        self.go_res_dist = go_res_dist
        self.go_backbone = go_backbone
        self.go_atomname = go_atomname
        self.molname = molname
        self.water_bias = water_bias
        self.water_bias_eps = water_bias_eps or []
        self.water_bias_idrs = water_bias_idrs or []

    def run_system(self, system):
        if system.go_params["go_map"]:
            LOGGER.info("Generating the Go model.", type="step")
            GoPipeline.run_system(system,
                                moltype=self.molname,
                                cutoff_short=self.go_low,
                                cutoff_long=self.go_up,
                                go_eps=self.go_eps,
                                res_dist=self.go_res_dist,
                                go_anchor_bead=self.go_backbone,
                                go_atomname=self.go_atomname)
            system.meta["defines"] = ("GO_VIRT",)
            system.meta["itp_paths"] = {"atomtypes": "go_atomtypes.itp","nonbond_params": "go_nbparams.itp"}
            if not self.water_bias:
                # this ensures that disordered-folded go bonds get removed regardless of force field.
                vermouth.processors.ComputeWaterBias(self.water_bias,
                                                    dict(self.water_bias_eps),
                                                    self.water_bias_idrs,
                                                    ).run_system(system)
        return system
    
class MergeChainsWrapper(Processor):
        def __init__(self, merge_chains = None):
            self.merge_chains = merge_chains
        def run_system(self, system):
            if not self.merge_chains:
                return system 
            #if all is not in the list of chains to be merged
            if "all" not in self.merge_chains:
                input_chain_sets = [i.split(",") for i in self.merge_chains]
                for chain_set in input_chain_sets:
                    vermouth.MergeChains(chains=chain_set, all_chains=False).run_system(system)
            #if all is in the list and is the only argument
            elif "all" in self.merge_chains and len(self.merge_chains) == 1:
                vermouth.MergeChains(chains=[], all_chains=True).run_system(system)
            #otherwise error because you cannot have all and specific chains at the same time.
            else:
                raise ValueError("Multiple conflicting merging arguments given. "
                                "Either specify -merge all or -merge A,B,C (+).")
            return system   
        
class ElasticWrapper(Processor):
    def __init__(
        self, 
        rb_force_constant,
        rb_lower_bound,
        rb_upper_bound,
        rb_decay_factor,
        rb_decay_power,
        rb_minimum_force,
        rb_selection,
        rb_unit,
        res_min_dist,
    ):
       self.rb_force_constant = rb_force_constant
       self.rb_lower_bound = rb_lower_bound
       self.rb_upper_bound = rb_upper_bound
       self.rb_decay_factor = rb_decay_factor
       self.rb_decay_power = rb_decay_power
       self.rb_minimum_force = rb_minimum_force
       self.rb_selection = rb_selection
       self.rb_unit = rb_unit
       self.res_min_dist = res_min_dist
    def run_system(self, system):
        LOGGER.info("Setting the rubber bands.", type="step")
        # if the rubber band unit is molecule, then the domain criterion is always true, no exclusion within the molecule.
        if self.rb_unit == "molecule":
            domain_criterion = vermouth.processors.apply_rubber_band.always_true
            # if all, then merge all molecules into one and apply rubber band to the whole system. 
        elif self.rb_unit == "all":
            vermouth.MergeAllMolecules().run_system(system)
            domain_criterion = vermouth.processors.apply_rubber_band.always_true
            # only beads in the same chain can be connected by rubber bands.
        elif self.rb_unit == "chain":
            domain_criterion = vermouth.processors.apply_rubber_band.same_chain
            # you can also choose your own region 
        else:
            regions = [
                tuple(int(i) for i in apair.split(":"))
                for apair in self.rb_unit.split(",")
            ] 
            # check if all regions are pairs and not more or less. 
            if any(len(region) != 2 for region in regions):
                message = (
                    'Faulty resid interval for elastic network unit: "{}".'.format(
                        self.rb_unit
                    )
                )
                LOGGER.critical(message)
                raise ValueError(message)
            # if that is not the case proceed. only within residue area. not between regions. 
            else:
                domain_criterion = (
                    vermouth.processors.apply_rubber_band.make_same_region_criterion(
                        regions
                    )
                )
        # check if the user has given a special selection for the rubber bands. 
        if self.rb_selection is not None:
            selector = functools.partial(
                selectors.proto_select_attribute_in,
                attribute="atomname",
                values=self.rb_selection,
            )
            # if not, use the backbone of the given force field.
        else:
            selector = functools.partial(selectors.select_backbone,
                                         bb_atomname=system.force_field.variables['bb_atomname'])
        rubber_band_processor = vermouth.ApplyRubberBand(
            lower_bound=self.rb_lower_bound,
            upper_bound=self.rb_upper_bound,
            decay_factor=self.rb_decay_factor,
            decay_power=self.rb_decay_power,
            base_constant=self.rb_force_constant,
            minimum_force=self.rb_minimum_force,
            selector=selector,
            domain_criterion=domain_criterion,
            res_min_dist=self.res_min_dist,
        )
        rubber_band_processor.run_system(system)
        return system

class WaterBiasWrapper(Processor):
    def __init__(self, water_bias, water_bias_eps, water_bias_idrs, go):
        self.water_bias = water_bias
        self.water_bias_eps = water_bias_eps or []
        self.water_bias_idrs = water_bias_idrs or []
        self.go = go
    def run_system(self, system):
        if not self.go:
            vermouth.rcsu.go_vs_includes.VirtualSiteCreator().run_system(system)
            # paths to the itp files for virtual sites. 
            itp_paths = {"atomtypes": "virtual_sites_atomtypes.itp",
                            "nonbond_params": "virtual_sites_nonbond_params.itp"}
        found_cgsecstruct = any(
            "cgsecstruct" in data
            for mol in system.molecules
            for _, data in mol.nodes(data=True)
        )

        print("cgsecstruct present:", found_cgsecstruct)
        # now we add a bias by defining specific virtual-site water interactions
        vermouth.processors.ComputeWaterBias(self.water_bias,
                                                    dict(self.water_bias_eps),
                                                    self.water_bias_idrs,
                                                    ).run_system(system)
        return system