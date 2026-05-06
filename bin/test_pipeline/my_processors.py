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
import networkx as nx
from vermouth.gmx.topology import write_gmx_topology
from vermouth.file_writer import DeferredFileWriter
from vermouth.log_helpers import ignore_warnings_and_count


VERSION = "martinize with vermouth {}".format(vermouth.__version__)
LOGGER = TypeAdapter(logging.getLogger("vermouth"))
# import the martinize2 classes en functions 

class WrapperMixin:
    def __init__(self, *args, **kwargs):
        args, kwargs = self.wrap(*args, **kwargs)
        super().__init__(*args, **kwargs)


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
        file_extension = self.path.suffix.upper()[1:]  # We do not keep the dot
        if file_extension in ["GRO"] and self.modelidx is not None:
            raise ValueError("GRO files don't know the concept of models.")
        if self.modelidx is None:
        # Set a sane default value. Can't do this using argparse machinery,
        # since we need to be able to check whether the flag was given.
            self.modelidx = 1
        # merge the lists of the ignored resnames into a set. 
        ignore_res = set()
        for grp in self.ignore_resnames:
            ignore_res.update(*grp)
        print(ignore_res)

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
    

class MakeBondsWrapper(WrapperMixin, vermouth.MakeBonds):
    @staticmethod
    def wrap(bonds_from="both", fudge=1.2):
        allow_name = bonds_from in ("name", "both")
        allow_dist = bonds_from in ("distance", "both")
        return (), {
            "allow_name": allow_name,
            "allow_dist": allow_dist,
            "fudge": fudge,
        }

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
class PrintFF(Processor):
    def __init__(self, force_field):
        self.force_field = force_field

    def run_system(self, system):
        print("FORCE FIELD:", self.force_field.name)
        return system
class AnnotateMutModWrapper(WrapperMixin, vermouth.AnnotateMutMod):
    @staticmethod
    def wrap(modify = None, cter = None, nter = None, mutate = None):
        modify = modify or [] # use an empty list if modify is None
        cter = cter or []
        nter = nter or []
        mutate = mutate or []

        # make the list which will contain all the differnet modifications 
        modifications = []
        # loop through the modify arguments and split them into two on the :, add them to the modifications list. 
        for item in modify:
            modifications.append(item.split(':'))

        for item in cter:
            modifications.append(['cter', item])

        for item in nter:
            modifications.append(['nter', item])

        # mutations list 
        mutations = []
        for items in mutate: 
            mutations.append(items.split(':'))

        # check if modifications is empty
        if modifications:
            # split all modifications into two lists. 
            resspecs, mods = zip(*modifications)
        else:
            # if there are no modifications, make empty lists for resspecs and mods.
            resspecs, mods = [], []

        # if no cter modification was given, add the default cter modification
        if not any("cter" in resspec for resspec in resspecs):
            modifications.append(["cter", "+C-ter"])

        if not any("nter" in resspec for resspec in resspecs):
            modifications.append(["nter", "+N-ter"])

        return(modifications, mutations), {}



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
class DSSPWrapper(WrapperMixin, vermouth.dssp.dssp.AnnotateDSSP):
    @staticmethod
    def wrap(executable=None, savedir="."):
        if not isinstance(executable, str):
            executable = None
        return (), {
            "executable": executable,
            "savedir": savedir,
        }

class SSWrapper(WrapperMixin, vermouth.dssp.dssp.AnnotateResidues):
    @staticmethod
    def wrap(ss=None):
        if ss is None:
            ss = ""
        return (), {
            "attribute": "aasecstruct",
            "sequence": ss.upper(),
            "molecule_selector": selectors.is_protein,
        }


class CollagenWrapper(WrapperMixin, vermouth.dssp.dssp.AnnotateResidues):
    @staticmethod
    def wrap():
        return (), {
            "attribute": "cgsecstruct",
            "sequence": "F",
            "molecule_selector": selectors.is_protein,
        }
    
class GoReader(Processor):
    def __init__(self, file_path):
        self.file_path = file_path
    def run_system(self, system):
            LOGGER.info("Reading Go model contact map.", type="step")
            read_go_map(system=system, file_path=self.file_path)
            return system
    
class ApplyPosresWrapper(WrapperMixin, vermouth.ApplyPosres):
    @staticmethod
    def wrap(posres, posres_fc, force_field):
        LOGGER.info("Applying position restraints.", type="step")
        node_selectors = {
            "all": (selectors.select_all, None),
            # look if the forcefield has the variable bb_atomname. the force_field object comes in the yaml from target_ff.
            "backbone": (
                selectors.select_backbone,
                force_field.variables["bb_atomname"]
            )   
        }
        node_selector = node_selectors[posres]
        return(node_selector, posres_fc), {}

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
        
class ElasticWrapper(WrapperMixin, vermouth.ApplyRubberBand):
    @staticmethod
    def wrap(
        rb_force_constant,
        rb_lower_bound,
        rb_upper_bound,
        rb_decay_factor,
        rb_decay_power,
        rb_minimum_force,
        rb_selection,
        rb_unit,
        res_min_dist,
        force_field,
    ):
        if rb_unit == "molecule":
            domain_criterion = vermouth.processors.apply_rubber_band.always_true

        elif rb_unit == "all":
            domain_criterion = vermouth.processors.apply_rubber_band.always_true

        elif rb_unit == "chain":
            domain_criterion = vermouth.processors.apply_rubber_band.same_chain

        else:
            regions = [
                tuple(int(i) for i in apair.split(":"))
                for apair in rb_unit.split(",")
            ]

            if any(len(region) != 2 for region in regions):
                raise ValueError(
                    f'Faulty resid interval for elastic network unit: "{rb_unit}".'
                )

            domain_criterion = (
                vermouth.processors.apply_rubber_band
                .make_same_region_criterion(regions)
            )

        if rb_selection is not None:
            selector = functools.partial(
                selectors.proto_select_attribute_in,
                attribute="atomname",
                values=rb_selection,
            )
        else:
            selector = functools.partial(
                selectors.select_backbone,
                bb_atomname=force_field.variables['bb_atomname'],
            )

        return (), {
            "lower_bound": rb_lower_bound,
            "upper_bound": rb_upper_bound,
            "decay_factor": rb_decay_factor,
            "decay_power": rb_decay_power,
            "base_constant": rb_force_constant,
            "minimum_force": rb_minimum_force,
            "selector": selector,
            "domain_criterion": domain_criterion,
            "res_min_dist": res_min_dist,
        }

class ComputeWaterBiasWrapper(WrapperMixin, vermouth.processors.ComputeWaterBias):
    @staticmethod
    def wrap(water_bias, water_bias_eps=None, water_bias_idrs=None):
        return (
            water_bias,
            dict(water_bias_eps or []),
            water_bias_idrs or [],
        ), {}
    

class OutputLoggerWrapper(Processor):
    def run_system(self, system):
        for molecule in system.molecules:
            LOGGER.debug("Writing molecule {}.", molecule, type="step")
            for loglevel, entries in molecule.log_entries.items():
                for entry, fmt_args in entries.items():
                    for fmt_arg in fmt_args:
                        fmt_arg = {str(k): molecule.nodes[v] for k, v in fmt_arg.items()}
                        LOGGER.log(loglevel, entry, **fmt_arg, type="model")
        return system

class OutputWriterWrapper(Processor):
    def __init__(self, top_path, outpath):
        self.top_path = top_path
        self.outpath = outpath
    def run_system(self, system):
        if self.top_path is not None:
            write_gmx_topology(system,
                           self.top_path,
                           itp_paths=system.meta.get("itp_paths", {}),
                           C6C12=False,
                           defines=system.meta.get("defines", ()),
                           ) 
        if self.outpath is not None:
            vermouth.pdb.write_pdb(system, str(self.outpath), omit_charges=True)


        return system
    
    
class ListBlocksWrapper(Processor):
    def __init__(self, from_ff, to_ff):
        self.from_ff = from_ff
        self.to_ff = to_ff

    def run_system(self, system):
        known_force_fields = vermouth.forcefield.find_force_fields(
            Path(DATA_PATH) / "force_fields"
        )

        if self.from_ff not in known_force_fields:
            raise ValueError(f'Unknown force field "{self.from_ff}".')
        if self.to_ff not in known_force_fields:
            raise ValueError(f'Unknown force field "{self.to_ff}".')

        print("The following Blocks are known to force field {}:".format(self.from_ff))
        print(", ".join(sorted(known_force_fields[self.from_ff].blocks)))
        print(
            "The following Modifications are known to force field {}:".format(
                self.from_ff
            )
        )
        print(", ".join(sorted(known_force_fields[self.from_ff].modifications)))
        print()

        print("The following Blocks are known to force field {}:".format(self.to_ff))
        print(", ".join(sorted(known_force_fields[self.to_ff].blocks)))
        print(
            "The following Modifications are known to force field {}:".format(
                self.to_ff
            )
        )
        print(", ".join(sorted(known_force_fields[self.to_ff].modifications)))

        raise SystemExit(0)