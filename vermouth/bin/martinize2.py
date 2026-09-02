"""Command-line entry point for the configurable Martinize2 pipeline."""

import logging
import sys
from pathlib import Path

import yaml

import vermouth
import vermouth.forcefield


from vermouth.log_helpers import TypeAdapter, StyleAdapter, BipolarFormatter, CountingHandler, ignore_warnings_and_count
from vermouth.file_writer import DeferredFileWriter
from vermouth import DATA_PATH
from vermouth.map_input import (
    read_mapping_directory,
    generate_all_self_mappings,
    combine_mappings,
)

from vermouth.pipeline import (
    build_mini_parser,
    PipelineConfigBuilder,
    CLIBuilder,
    PipelineBuilder,
    find_step_by_id,
    insert_pipeline_step,
    merge_override,
)
# logging.basicConfig(level=logging.INFO)
LOGGER = TypeAdapter(logging.getLogger("vermouth"))

PRETTY_FORMATTER = logging.Formatter(
    fmt="{levelname:>8} - {type} - {message}", style="{"
)
DETAILED_FORMATTER = logging.Formatter(
    fmt="{levelname:>8} - {type} - {name} - {message}", style="{"
)

COUNTER = CountingHandler()

# Control above what level message we want to count
COUNTER.setLevel(logging.WARNING)

CONSOLE_HANDLER = logging.StreamHandler()
FORMATTER = BipolarFormatter(
    DETAILED_FORMATTER, PRETTY_FORMATTER, logging.DEBUG, logger=LOGGER
)
CONSOLE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(CONSOLE_HANDLER)
LOGGER.addHandler(COUNTER)

LOGGER = StyleAdapter(LOGGER)



def force_fields(args, parser):
    """
    Load force fields and mappings used by Martinize2.

    Force fields and mappings are loaded from the default Vermouth data
    directories and from any additional directories provided through the
    command-line interface. Self-mappings are generated for all known force
    fields.

    Parameters
    ----------
    args : dict
        Parsed command-line arguments.
    parser : argparse.ArgumentParser
        Argument parser used to exit after listing available force fields.

    Returns
    -------
    tuple[dict, dict]
        The known force fields and available mappings.
    """ 
    known_force_fields = vermouth.forcefield.find_force_fields(
        Path(DATA_PATH) / "force_fields"
    )

    known_mappings = read_mapping_directory(
        Path(DATA_PATH) / "mappings", known_force_fields
    )

    for directory in args["extra_ff_dir"]:
        vermouth.forcefield.find_force_fields(directory, known_force_fields)

    for directory in args["extra_map_dir"]:
        partial_mapping = read_mapping_directory(directory, known_force_fields)
        combine_mappings(known_mappings, partial_mapping)

    if args["list_ff"]:
        print("The following force fields are known:")
        for idx, ff_name in enumerate(reversed(list(known_force_fields)), 1):
            print("{:3d}. {}".format(idx, ff_name))
        parser.exit()

    partial_mapping = generate_all_self_mappings(known_force_fields.values())
    combine_mappings(known_mappings, partial_mapping)

    return known_force_fields, known_mappings


def main():
    """
    Build and run the configured Martinize2 pipeline.

    The function loads the selected pipeline configuration, applies optional
    overrides, builds the dynamic command-line interface, resolves force
    fields and mappings, constructs the pipeline, and runs it on a molecular
    system.
    """
    mini_parser = build_mini_parser()
    mini_args, remaining_args = mini_parser.parse_known_args()

    override_conf = None

    if mini_args.override is not None:
        with open(mini_args.override, "r", encoding="utf-8") as file:
            override_conf = yaml.safe_load(file)

    loglevels = {0: logging.INFO, 1: logging.DEBUG, 2: 5}
    LOGGER.setLevel(loglevels[mini_args.verbosity])

    config_builder = PipelineConfigBuilder(
        mini_args.pipeline,
        mini_args.pipeline_dir,
    )
    configs, pipeline_conf = config_builder.build_config()

    
    if override_conf is not None:
        overrides = override_conf.get("override", {})

        for step_id, changes in overrides.items():
            step = find_step_by_id(pipeline_conf, step_id, raise_if_missing=False)

            if step is not None:
                merge_override(step, changes)
            else:
                insert_pipeline_step(
                    pipeline_conf,
                    {
                        **changes,
                        "id": step_id,
                    },
                )
    cli_builder = CLIBuilder(pipeline_conf)
    cli_builder.build_argparser()
    parser = cli_builder.argparser
    cli_args = cli_builder.parse_cli_args(remaining_args)
    

    cli_args.update(vars(mini_args))
    known_force_fields, mappings = force_fields(cli_args, parser)

    variables = {}

    for namespace, conf in configs:
        root = conf["martinize2"]

        if "ff" in root.get("variables", []):
            if "from_ff" in root.get("cli_flags", {}):
                variables[f"{namespace}.ff"] = known_force_fields[cli_args["from_ff"]]

            elif "to_ff" in root.get("cli_flags", {}):
                variables[f"{namespace}.ff"] = known_force_fields[cli_args["to_ff"]]

            else:
                variables[f"{namespace}.ff"] = known_force_fields[namespace]

        if "mappings" in root.get("variables", []):
            variables[f"{namespace}.mappings"] = mappings

    pipeline_builder = PipelineBuilder(pipeline_conf)
    pipeline = pipeline_builder.build_pipeline(cli_args, variables)



    source_ff = known_force_fields[cli_args["from_ff"]]
    system = vermouth.System(force_field=source_ff)

    pipeline.run_system(system)

    leftover_warnings = ignore_warnings_and_count(COUNTER, cli_args["maxwarn"])

    if leftover_warnings:
        LOGGER.error(
            "%s warnings were encountered after accounting for the "
            "-maxwarn flag. No output files will be "
            "written. Consider fixing the warnings, or if you are sure "
            "they are harmless, use the -maxwarn flag.",
            leftover_warnings,
        )
        sys.exit(2)

    DeferredFileWriter().write()
    vermouth.Quoter().run_system(system)


def entry():
    """Run the Martinize2 command-line interface."""
    main()


if __name__ == "__main__":
    entry()