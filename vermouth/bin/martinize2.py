from pathlib import Path

import yaml
import vermouth
import vermouth.forcefield
import sys
import logging

from vermouth.log_helpers import CountingHandler, ignore_warnings_and_count
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
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("vermouth")
COUNTER = CountingHandler()
COUNTER.setLevel(logging.WARNING)
LOGGER.addHandler(COUNTER)


def force_fields(args, parser):
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
    mini_parser = build_mini_parser()
    mini_args, remaining_args = mini_parser.parse_known_args()

    override_conf = None

    if mini_args.override is not None:
        with open(mini_args.override, "r", encoding="utf-8") as file:
            override_conf = yaml.safe_load(file)

        

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
    parser = cli_builder.build_argparser()
    cli_args = cli_builder.parse_cli_args(remaining_args)

    cli_args.update(vars(mini_args))

    known_force_fields, mappings = force_fields(cli_args, parser)

    variables = {}
    for namespace, conf in configs:
        root = conf["martinize2"]

        if namespace in known_force_fields:
            variables[f"{namespace}.ff"] = known_force_fields[namespace]

        if "mappings" in root.get("variables", []):
            variables[f"{namespace}.mappings"] = mappings

    pipeline_builder = PipelineBuilder(pipeline_conf)
    pipeline = pipeline_builder.build_pipeline(cli_args, variables)



    first_ff_name = configs[0][0]
    source_ff = known_force_fields[first_ff_name]
    system = vermouth.System(force_field=source_ff)

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

    DeferredFileWriter().write()
    vermouth.Quoter().run_system(system)
    print(system.meta.get("header"))


def entry():
    main()


if __name__ == "__main__":
    entry()