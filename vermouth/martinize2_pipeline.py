"""Martinize2-specific configuration built on the generic pipeline engine."""

import argparse
from collections import OrderedDict
from pathlib import Path

import vermouth
from vermouth.pipeline import (
    compose_pipeline_config,
    find_pipeline_configs,
    validate_cli_options,
)


def _cys_argument(value):
    """Parse a cysteine bridge command-line argument."""
    try:
        return float(value)
    except ValueError:
        match value.lower():
            case "auto" | "none" as parsed:
                return parsed
            case _:
                raise argparse.ArgumentTypeError(
                    'Value must be "auto", "none", or a float.'
                )


def water_bias(value):
    """Parse a water-bias residue/epsilon command-line argument."""
    try:
        letter, epsilon = value.split(":")
        return letter, float(epsilon)
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "value must be a letter and a float separated by a colon"
        ) from error


def ignore_resname(value):
    """Parse a comma-separated list of residue names."""
    return [item.strip() for item in value.split(",") if item.strip()]


def maxwarn(value):
    """Parse a warning type and optional allowed warning count."""
    message = (
        "Values for the -maxwarn option must be the name of a warning type, "
        "a number, or following the format '<warning-type>:<count>' where "
        "<warning-type> is the name of the warning type to ignore, and "
        "<count> is the number of warning of that type to ignore. "
        f"'{value}' is not a valid value."
    )
    split_value = value.split(":")
    if len(split_value) == 1:
        try:
            return None, int(value)
        except ValueError:
            return value, None
    if len(split_value) == 2:
        try:
            return split_value[0], int(split_value[1])
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(message)


TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "path": Path,
    "cys_argument": _cys_argument,
    "water_bias": water_bias,
    "ignore_resname": ignore_resname,
    "maxwarn": maxwarn,
}


class _MiniArgumentParser(argparse.ArgumentParser):
    """Defer abbreviated single-dash options to the full CLI parser."""

    def _get_option_tuples(self, option_string):
        if (
            option_string.startswith("-")
            and not option_string.startswith("--")
            and option_string not in self._option_string_actions
            and any(
                option.startswith(option_string)
                for option in self._option_string_actions
            )
        ):
            return []
        return super()._get_option_tuples(option_string)


def build_mini_parser():
    """Build the preliminary parser used to select Martinize2 pipelines."""
    parser = _MiniArgumentParser(add_help=False, allow_abbrev=False)
    force_field_group = parser.add_argument_group("Force field selection")
    force_field_group.add_argument(
        "-from",
        dest="from_ff",
        default="charmm",
        help="Source force field and pipeline name.",
    )
    force_field_group.add_argument(
        "-ff",
        dest="to_ff",
        default="martini3001",
        help="Target force field and pipeline name.",
    )
    force_field_group.add_argument(
        "-ff-dir",
        "-extra_ff_dir",
        dest="extra_ff_dir",
        action="append",
        default=[],
        type=Path,
        help="Additional repository for custom force fields.",
    )
    force_field_group.add_argument(
        "-map-dir",
        "-extra_map_dir",
        dest="extra_map_dir",
        action="append",
        default=[],
        type=Path,
        help="Additional repository for mapping files.",
    )
    force_field_group.add_argument(
        "-list-ff",
        "-list_ff",
        dest="list_ff",
        action="store_true",
        help="List all known force fields, and exit.",
    )
    parser.add_argument(
        "-pipeline",
        nargs="+",
        default=[],
        help="Additional pipeline YAML fragments to include.",
    )
    parser.add_argument(
        "-pipeline-dir",
        action="append",
        default=[],
        type=Path,
        help="Directory to search pipeline YAML files in.",
    )
    parser.add_argument(
        "-v",
        dest="verbosity",
        action="count",
        help="Enable debug logging output. Can be given multiple times.",
        default=0,
    )
    parser.add_argument(
        "-maxwarn",
        dest="maxwarn",
        type=maxwarn,
        action="append",
        nargs="+",
        default=[],
        help="The maximum number of allowed warnings.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="martinize2",
    )
    return parser


def build_pipeline_document(
    from_pipeline,
    to_pipeline,
    pipeline_dirs=(),
    additional_pipelines=(),
):
    """Build the Martinize2 document composed from source and target pipelines."""
    document = {
        "$schema": "./pipeline-schema.yaml",
        "martinize2": {
            "$include": [
                *(f"{pipeline}:martinize2" for pipeline in additional_pipelines),
            ],
            "steps": OrderedDict([
                (
                    "from",
                    {
                        "$include": [
                            {
                                "path": f"{from_pipeline}:martinize2.steps.from",
                                "rename_variables": {"ff": "source_ff"},
                            },
                        ],
                    },
                ),
                (
                    "to",
                    {
                        "$include": [f"{to_pipeline}:martinize2.steps.to"],
                    },
                ),
            ]),
        },
    }
    generated_path = vermouth.DATA_PATH / "pipelines" / "generated-pipeline.yaml"
    document = compose_pipeline_config(document, generated_path, pipeline_dirs)
    validate_cli_options(document["martinize2"], path="martinize2")
    return document


class PipelineConfigBuilder:
    """Build the Martinize2 pipeline configuration from YAML fragments."""

    def __init__(
        self,
        from_pipeline,
        to_pipeline,
        pipeline_names=(),
        pipeline_dirs=(),
    ):
        self.pipeline_names = pipeline_names
        self.pipeline_dirs = pipeline_dirs
        self.from_pipeline = from_pipeline
        self.to_pipeline = to_pipeline
        self.paths = []

    def build_config(self):
        """Build the document for the selected source and target pipelines."""
        selected_pipelines = [
            self.from_pipeline,
            self.to_pipeline,
            *self.pipeline_names,
        ]
        self.paths = list(
            find_pipeline_configs(selected_pipelines, self.pipeline_dirs)
        )
        return build_pipeline_document(
            self.paths[0],
            self.paths[1],
            self.pipeline_dirs,
            self.pipeline_names,
        )
