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

import logging
import sys
from pathlib import Path

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

    The function loads the selected pipeline configuration, builds the dynamic
    command-line interface, resolves force
    fields and mappings, constructs the pipeline, and runs it on a molecular
    system.
    """
    mini_parser = build_mini_parser()
    mini_args, remaining_args = mini_parser.parse_known_args()

    loglevels = {0: logging.INFO, 1: logging.DEBUG, 2: 5}
    LOGGER.setLevel(loglevels[mini_args.verbosity])

    known_force_fields, mappings = force_fields(vars(mini_args), mini_parser)
    try:
        source_force_field = known_force_fields[mini_args.from_ff]
        target_force_field = known_force_fields[mini_args.to_ff]
    except KeyError as error:
        mini_parser.error(f"Unknown force field: {error.args[0]!r}")

    config_builder = PipelineConfigBuilder(
        mini_args.pipeline,
        mini_args.pipeline_dir,
        source_force_field,
        target_force_field,
    )
    configs, pipeline_document = config_builder.build_config()
    pipeline_conf = pipeline_document["martinize2"]

    cli_builder = CLIBuilder('martinize2', pipeline_conf)
    config_paths = []
    for path in config_builder.paths:
        try:
            config_paths.append(path.relative_to(Path.cwd()))
        except ValueError:
            config_paths.append(path)

    cli_builder.build_argparser(
        parents=[mini_parser],
        added_flags={"from_ff", "to_ff"},
        epilog=f'Pipeline and CLI built from {', '.join(str(p) for p in config_paths)}',
    )
    parser = cli_builder.argparser
    cli_args = cli_builder.parse_cli_args(remaining_args)


    cli_args.update(vars(mini_args))
    variables = {
        "source_ff": source_force_field,
        "ff": target_force_field,
        "mappings": mappings,
    }

    pipeline_builder = PipelineBuilder(pipeline_conf)
    pipeline = pipeline_builder.build_pipeline(cli_args, variables)


    system = vermouth.System(force_field=source_force_field)

    pipeline.run_system(system)

    leftover_warnings = ignore_warnings_and_count(COUNTER, cli_args["maxwarn"])

    if leftover_warnings:
        LOGGER.error(
            "{} warnings were encountered after accounting for the "
            "-maxwarn flag. No output files will be "
            "written. Consider fixing the warnings, or if you are sure "
            "they are harmless, use the -maxwarn flag.",
            leftover_warnings,
        )
        sys.exit(2)
    else:
        DeferredFileWriter().write()
        vermouth.Quoter().run_system(system)


def entry():
    """Run the Martinize2 command-line interface."""
    main()


if __name__ == "__main__":
    entry()