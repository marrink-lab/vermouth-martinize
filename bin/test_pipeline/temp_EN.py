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