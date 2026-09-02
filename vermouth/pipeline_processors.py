import vermouth
from vermouth.processors.processor import Processor
import vermouth.pdb
import logging
from vermouth.log_helpers import TypeAdapter
import functools
from vermouth import selectors
from vermouth.rcsu.go_pipeline import GoPipeline
from vermouth.gmx.topology import write_gmx_topology
from vermouth.rcsu.contact_map import read_go_map


VERSION = "martinize with vermouth {}".format(vermouth.__version__)
LOGGER = TypeAdapter(logging.getLogger("vermouth"))
# import the martinize2 classes en functions 

class WrapperMixin:
    """
    Adapt wrapper arguments before initializing the parent processor.

    Subclasses implement ``wrap`` to translate pipeline arguments to the
    positional and keyword arguments expected by the wrapped processor.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the wrapped processor with translated arguments.

        Parameters
        ----------
        *args
            Positional arguments passed to ``wrap``.
        **kwargs
            Keyword arguments passed to ``wrap``.
        """
        args, kwargs = self.wrap(*args, **kwargs)
        super().__init__(*args, **kwargs)


def read_system(system, path, ignore_resnames=(), ignh=None, modelidx=None):
    """
    Read molecular coordinates into a system.

    The input format is determined from the file extension. PDB, ENT, GRO,
    and CIF files are supported.

    Parameters
    ----------
    system : vermouth.system.System
        System into which the molecular structure is read.
    path : pathlib.Path
        Path to the input structure file.
    ignore_resnames : iterable[str], optional
        Residue names to exclude while reading.
    ignh : bool, optional
        Whether hydrogen atoms should be ignored.
    modelidx : int, optional
        Model index to read from formats that support multiple models.

    Returns
    -------
    vermouth.system.System
        System containing the structures read from the input file.

    Raises
    ------
    ValueError
        If the input file has an unsupported extension.
    """
    file_extension = path.suffix.upper()[1:]  # We do not keep the dot
    if file_extension in ["PDB", "ENT"]:
        vermouth.PDBInput(str(path), exclude=ignore_resnames, ignh=ignh, modelidx=modelidx).run_system(system)
    elif file_extension in ["GRO"]:
        vermouth.GROInput(str(path), exclude=ignore_resnames, ignh=ignh).run_system(system)
    elif file_extension in ["CIF"]:
        vermouth.CIFInput(str(path), exclude=ignore_resnames, ignh=ignh,modelidx=modelidx).run_system(system)
    else:
        raise ValueError('Unknown file extension "{}".'.format(file_extension))
    return system


# define the processor class. readsystem itself does not have a forcefield. 
class ReadSystem(Processor):
    """
    Read the input molecular structure into a Vermouth system.

    Parameters
    ----------
    path : pathlib.Path
        Path to the input structure file.
    ignore_resnames : iterable, optional
        Residue names that should be ignored.
    ignh : bool, optional
        Whether hydrogen atoms should be ignored.
    modelidx : int, optional
        Model index to read.
    """
    def __init__(self, path, ignore_resnames=(), ignh=None, modelidx=None):
        self.path = path
        # ignore resnames is a list of lists. it needs to be a set of strings.
        ignore_res = set()
        for group in ignore_resnames:
            ignore_res.update(*group)

        self.ignore_resnames = ignore_res
        self.ignh = ignh
        self.modelidx = modelidx
    # define run system 
    def run_system(self, system):
        """
        Read the configured input structure into a system.

        Parameters
        ----------
        system : vermouth.system.System
            System into which the molecular structure is read.

        Returns
        -------
        vermouth.system.System
            Updated system.

        Raises
        ------
        ValueError
            If a model index is specified for a GRO file.
        """
        LOGGER.info("Running ReadSystem processor", type="step")
        LOGGER.info("Input file: %s", self.path)
        file_extension = self.path.suffix.upper()[1:]  # We do not keep the dot
        if file_extension in ["GRO"] and self.modelidx is not None:
            raise ValueError("GRO files don't know the concept of models.")
        if self.modelidx is None:
        # Set a sane default value. Can't do this using argparse machinery,
        # since we need to be able to check whether the flag was given.
            self.modelidx = 1
        

        system = read_system(
            system=system,
            path=self.path,
            ignore_resnames=self.ignore_resnames,
            ignh=self.ignh,
            modelidx=self.modelidx,
        )
        
        return system
    

class MakeBondsWrapper(WrapperMixin, vermouth.MakeBonds):
    """Adapt pipeline options for the ``MakeBonds`` processor."""
    @staticmethod
    def wrap(bonds_from="both", fudge=1.2):
        """
        Translate bond detection options to ``MakeBonds`` arguments.

        Parameters
        ----------
        bonds_from : str, optional
            Method used to determine bonds.
        fudge : float, optional
            Distance scaling factor used for bond detection.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``MakeBonds``.
        """
        allow_name = bonds_from in ("name", "both")
        allow_dist = bonds_from in ("distance", "both")
        return (), {
            "allow_name": allow_name,
            "allow_dist": allow_dist,
            "fudge": fudge,
        }

class AnnotateMutModWrapper(WrapperMixin, vermouth.AnnotateMutMod):
    """Adapt mutation and modification options for ``AnnotateMutMod``."""
    @staticmethod
    def wrap(modify =None, cter =None, nter =None, mutate =None, neutral_termini=False,):
        """
        Convert CLI mutation and modification options.

        Parameters
        ----------
        modify : iterable[str], optional
            Residue modifications.
        cter : iterable[str], optional
            C-terminal modifications.
        nter : iterable[str], optional
            N-terminal modifications.
        mutate : iterable[str], optional
            Residue mutations.
        neutral_termini : bool, optional
            Use neutral terminal modifications.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``AnnotateMutMod``.
        """
        modify = modify or [] # use an empty list if modify is None
        cter = cter or []
        nter = nter or []
        mutate = mutate or []
        neutral_termini = neutral_termini or False
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


        if neutral_termini:
            if not any("cter" in resspec for resspec in resspecs):
                modifications.append(["cter", "+COOH-ter"])
            if not any("nter" in resspec for resspec in resspecs):
                modifications.append(["nter", "+NH2-ter"])
        else:        
            # if no cter modification was given, add the default cter modification
            if not any("cter" in resspec for resspec in resspecs):
                modifications.append(["cter", "+C-ter"])

            if not any("nter" in resspec for resspec in resspecs):
                modifications.append(["nter", "+N-ter"])

        return(modifications, mutations), {}
 
# secundairy structure options. you can give only 1 out of 3. 
class DSSPWrapper(WrapperMixin, vermouth.dssp.dssp.AnnotateDSSP):
    """Adapt DSSP configuration options for ``AnnotateDSSP``."""
    @staticmethod
    def wrap(executable=None, savedir="."):
        """
        Translate DSSP options.

        Parameters
        ----------
        executable : str, optional
            DSSP executable to use.
        savedir : str or pathlib.Path, optional
            Directory in which DSSP output is stored.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``AnnotateDSSP``.
        """
        if not isinstance(executable, str):
            executable = None
        return (), {
            "executable": executable,
            "savedir": savedir,
        }

class SSWrapper(WrapperMixin, vermouth.dssp.dssp.AnnotateResidues):
    """Adapt manual secondary-structure options for ``AnnotateResidues``."""
    @staticmethod
    def wrap(attr, seq=""):
        """
        Translate secondary-structure annotation options.

        Parameters
        ----------
        attr : str
            Residue attribute in which the annotation is stored.
        seq : str, optional
            Secondary-structure sequence.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``AnnotateResidues``.
        """
        return (), {
            "attribute": attr,
            "sequence": seq.upper(),
            "molecule_selector": selectors.is_protein,
        }
    
class GoReader(Processor):
    """
    Read a Go-model contact map.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the Go-model contact map.
    """
    def __init__(self, file_path):
        self.file_path = file_path
    def run_system(self, system):
        """
        Read the configured Go contact map into a system.

        Parameters
        ----------
        system : vermouth.system.System
            System to update.

        Returns
        -------
        vermouth.system.System
            Updated system.
        """
        LOGGER.info("Reading Go model contact map.", type="step")
        read_go_map(system=system, file_path=self.file_path)
        return system
    
class ApplyPosresWrapper(WrapperMixin, vermouth.ApplyPosres):
    """Adapt position-restraint options for ``ApplyPosres``."""
    @staticmethod
    def wrap(posres, posres_fc, force_field):
        """
        Build the selector and force constant for position restraints.

        Parameters
        ----------
        posres : str
            Atom selection used for position restraints.
        posres_fc : float
            Position-restraint force constant.
        force_field : vermouth.forcefield.ForceField
            Force field used to determine backbone atoms.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``ApplyPosres``.
        """
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
    """
    Generate Go-model interactions for a molecular system.

    Parameters
    ----------
    go_low : float
        Lower contact-distance cutoff.
    go_up : float
        Upper contact-distance cutoff.
    go_eps : float
        Interaction strength.
    go_res_dist : int
        Minimum residue separation.
    go_backbone : str
        Backbone bead used as the Go-model anchor.
    go_atomname : str
        Atom name used for generated Go virtual sites.
    molname : str
        Molecule type name.
    water_bias : bool, optional
        Whether water-bias interactions are enabled.
    water_bias_eps : iterable, optional
        Water-bias epsilon values.
    water_bias_idrs : iterable, optional
        Regions used for water-bias handling.
    """
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
        """
        Generate Go-model interactions when a contact map is available.

        Parameters
        ----------
        system : vermouth.system.System
            System to process.

        Returns
        -------
        vermouth.system.System
            Processed system.
        """
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
    """
    Merge selected molecular chains.

    Parameters
    ----------
    merge_chains : iterable[str], optional
        Chain groups to merge, or ``"all"`` to merge all chains.
    """
    def __init__(self, merge_chains = None):
        self.merge_chains = merge_chains
    def run_system(self, system):
        """
        Merge the configured chains.

        Parameters
        ----------
        system : vermouth.system.System
            System to process.

        Returns
        -------
        vermouth.system.System
            Processed system.

        Raises
        ------
        ValueError
            If ``all`` is combined with specific chain selections.
        """
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
    """Adapt elastic-network options for ``ApplyRubberBand``."""
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
        """
        Translate elastic-network options.

        Parameters
        ----------
        rb_force_constant : float
            Base elastic-network force constant.
        rb_lower_bound : float
            Lower distance cutoff.
        rb_upper_bound : float
            Upper distance cutoff.
        rb_decay_factor : float
            Force decay factor.
        rb_decay_power : float
            Force decay power.
        rb_minimum_force : float
            Minimum elastic-network force.
        rb_selection : iterable[str] or None
            Atom names included in the elastic network.
        rb_unit : str
            Unit within which elastic interactions are generated.
        res_min_dist : int
            Minimum residue separation.
        force_field : vermouth.forcefield.ForceField
            Force field used to determine backbone atoms.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``ApplyRubberBand``.

        Raises
        ------
        ValueError
            If a custom residue interval is invalid.
        """
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
    """Adapt water-bias options for ``ComputeWaterBias``."""
    @staticmethod
    def wrap(water_bias, water_bias_eps=None, water_bias_idrs=None):
        """
        Translate water-bias options.

        Parameters
        ----------
        water_bias : bool
            Whether water bias is enabled.
        water_bias_eps : iterable, optional
            Residue-specific epsilon values.
        water_bias_idrs : iterable, optional
            Intrinsically disordered regions.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``ComputeWaterBias``.
        """
        return (
            water_bias,
            dict(water_bias_eps) or [],
            water_bias_idrs or [],
        ), {}

class OutputWriterWrapper(Processor):
    """
    Write topology and coordinate output files.

    Parameters
    ----------
    top_path : str or pathlib.Path or None
        Path of the topology output file.
    outpath : str or pathlib.Path or None
        Path of the PDB output file.
    """
    def __init__(self, top_path, outpath):
        self.top_path = top_path
        self.outpath = outpath
    def run_system(self, system):
        """
        Write configured output files.

        Parameters
        ----------
        system : vermouth.system.System
            System to write.

        Returns
        -------
        vermouth.system.System
            Unmodified input system.
        """
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
    
    
class ListBlocks(Processor):
    """
    Print known blocks and modifications for two force fields.

    Parameters
    ----------
    from_ff : vermouth.forcefield.ForceField
        Source force field.
    to_ff : vermouth.forcefield.ForceField
        Target force field.
    """
    def __init__(self, from_ff, to_ff):
        self.from_ff = from_ff
        self.to_ff = to_ff
    def run_system(self, _system):
        """
        Print available blocks and modifications and terminate execution.

        Parameters
        ----------
        _system : vermouth.system.System
            Current molecular system.

        Raises
        ------
        SystemExit
            Always raised after the force-field information is printed.
        """
        print(f"The following Blocks are known to force field {self.from_ff.name}:")
        print(", ".join(sorted(self.from_ff.blocks)))

        print(f"The following Modifications are known to force field {self.from_ff.name}:")
        print(", ".join(sorted(self.from_ff.modifications)))
        print()

        print(f"The following Blocks are known to force field {self.to_ff.name}:")
        print(", ".join(sorted(self.to_ff.blocks)))

        print(f"The following Modifications are known to force field {self.to_ff.name}:")
        print(", ".join(sorted(self.to_ff.modifications)))

        raise SystemExit(0)

class SetMoleculeMetaScFixWrapper(WrapperMixin, vermouth.SetMoleculeMeta):
    """Adapt side-chain-fixing options for ``SetMoleculeMeta``."""
    @staticmethod
    def wrap(noscfix=False):
        """
        Convert the ``noscfix`` option to molecule metadata.

        Parameters
        ----------
        noscfix : bool, optional
            Disable side-chain fixing when true.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``SetMoleculeMeta``.
        """
        return (), {
            "scfix": not noscfix,
        }
    
class NameMolTypeWrapper(WrapperMixin, vermouth.NameMolType):
    """Adapt molecule naming options for ``NameMolType``."""
    @staticmethod
    def wrap(keep_duplicate_itp=False, molname=None):
        """
        Translate molecule naming and deduplication options.

        Parameters
        ----------
        keep_duplicate_itp : bool, optional
            Keep duplicate molecule topology definitions.
        molname : str, optional
            Molecule type name.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``NameMolType``.
        """
        return (), {
            "deduplicate": not keep_duplicate_itp,
            "molname": molname,
        }

class VirtualSiteCreatorWrapper(vermouth.rcsu.go_vs_includes.VirtualSiteCreator):
    """Create Go virtual sites and register their topology include files."""
    def run_system(self, system):
        """
        Create virtual sites and update topology include metadata.

        Parameters
        ----------
        system : vermouth.system.System
            System to process.

        Returns
        -------
        vermouth.system.System
            Processed system.
        """
        result = super().run_system(system)

        system.meta["itp_paths"] = {
            "atomtypes": "virtual_sites_atomtypes.itp",
            "nonbond_params": "virtual_sites_nonbond_params.itp",
        }

        return result if result is not None else system

class SetMoleculeMetaIDRWrapper(WrapperMixin, vermouth.SetMoleculeMeta):
    """Adapt IDR options for ``SetMoleculeMeta``."""
    @staticmethod
    def wrap(id_regions=None):
        """
        Convert IDR region configuration to molecule metadata.

        Parameters
        ----------
        id_regions : iterable, optional
            Configured intrinsically disordered regions.

        Returns
        -------
        tuple
            Positional and keyword arguments for ``SetMoleculeMeta``.
        """
        return (), {
            "idr": bool(id_regions),
        }