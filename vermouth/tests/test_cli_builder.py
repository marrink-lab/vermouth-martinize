"""
Tests for the CLIBuilder class.
"""

import sys
from pathlib import Path
from collections import OrderedDict
sys.path.insert(0, str(Path(__file__).parent))
import pytest
from vermouth.pipeline import CLIBuilder



@pytest.fixture
def simple_pipeline_conf():
    """
    A simple pipeline config with only CLI flags and no steps.
    """
    return {
        "cli": {
            "flags": {
                "inpath": {
                    "type": "path",
                    "required": True,
                },
                "elastic": {
                    "action": "store_true",
                    "default": False,
                },
                "maxwarn": {
                    "type": "int",
                    "default": 0,
                },
            },
        },
        "steps": OrderedDict(),
    }

def test_inpath_flag(simple_pipeline_conf):
    """
    Test that CLIBuilder can correctly parse a simple CLI flag.
    """

    cli_builder = CLIBuilder("test", simple_pipeline_conf)

    args = cli_builder.parse_cli_args(['-inpath', 'test.pdb'])

    assert args['inpath'] == Path('test.pdb')

def test_default_value(simple_pipeline_conf):
    """
    Test that CLIBuilder can correctly handle default values.
    """
    
    cli_builder = CLIBuilder("test", simple_pipeline_conf)

    args = cli_builder.parse_cli_args(["-inpath", "test.pdb"])

    assert args["elastic"] is False

def test_store_true_flag(simple_pipeline_conf):
    """
    Test that CLIBuilder can correctly parse a store_true CLI flag.
    """

    cli_builder = CLIBuilder("test", simple_pipeline_conf)

    args = cli_builder.parse_cli_args([
        "-inpath", "test.pdb",
        "-elastic",
    ])

    assert args["elastic"] is True

def test_int_type(simple_pipeline_conf):
    """
    Test that CLIBuilder can correctly parse an integer CLI flag.
    """

    cli_builder = CLIBuilder("test", simple_pipeline_conf)

    args = cli_builder.parse_cli_args([
        "-inpath", "test.pdb",
        "-maxwarn", "3",
    ])

    assert args["maxwarn"] == 3


@pytest.fixture
def nested_pipeline_conf():
    """
    A pipeline config with nested steps and CLI flags at different levels.
    """

    return {
        "steps": OrderedDict([
            (
                "group",
                {
                    "cli": {
                        "flags": {
                            "molname": {
                                "type": "str",
                                "default": "molecule",
                            },
                        },
                    },
                    "steps": OrderedDict(),
                },
            ),
        ]),
    }


def test_nested_cli_flags(nested_pipeline_conf):
    """
    Test that CLIBuilder can correctly parse CLI flags in nested steps.
    """

    cli_builder = CLIBuilder("test", nested_pipeline_conf)

    args = cli_builder.parse_cli_args(["-molname", "protein 1"])

    assert args["molname"] == "protein 1"

def test_duplicate_cli_flag_is_added_once():
    """
    Test that duplicate CLI flags are not added multiple times.
    """
    pipeline_conf = {
        "cli": {
            "flags": {
                "inpath": {
                    "type": "path",
                    "required": True,
                },
            },
        },
        "steps": OrderedDict([
            (
                "nested_step",
                {
                    "cli": {
                        "flags": {
                            "inpath": {
                                "type": "path",
                                "required": True,
                            },
                        },
                    },
                    "steps": OrderedDict(),
                },
            ),
        ]),
    }

    cli_builder = CLIBuilder("test", pipeline_conf)
    parser = cli_builder.argparser

    inpath_actions = [
        action
        for action in parser._actions
        if "-inpath" in action.option_strings
    ]

    assert len(inpath_actions) == 1

def test_mutually_exclusive_cli_group():
    """
    Test that CLIBuilder creates a mutually exclusive CLI group.
    """
    pipeline_conf = {
        "cli": {
            "exclusive_groups": [
            {
                "flags": {
                    "dssp": {
                        "action": "store_true",
                        "default": False,
                    },
                    "ss": {
                        "action": "store_true",
                        "default": False,
                    },
                },
            }
            ],
        },
        "steps": OrderedDict(),
    }

    cli_builder = CLIBuilder("test", pipeline_conf)

    with pytest.raises(SystemExit):
        cli_builder.parse_cli_args([
            "-dssp",
            "-ss",
        ])

def test_unknown_cli_type_raises_error():
    """
    Test that an unknown CLI type raises a ValueError.
    """
    pipeline_conf = {
        "cli": {
            "flags": {
                "bad_flag": {
                    "type": "unknown_type",
                },
            },
        },
        "steps": OrderedDict(),
    }

    cli_builder = CLIBuilder("test", pipeline_conf)

    with pytest.raises(ValueError):
        cli_builder.build_argparser()

def test_empty_cli_group_is_skipped():
    """
    Test that CLIBuilder skips a CLI group when all group flags already exist.
    """
    pipeline_conf = {
        "cli": {
            "flags": {
                "dssp": {
                    "action": "store_true",
                    "default": False,
                },
                "ss": {
                    "action": "store_true",
                    "default": False,
                },
            },
            "exclusive_groups": [
                {
                    "flags": {
                        "dssp": {
                            "action": "store_true",
                            "default": False,
                        },
                        "ss": {
                            "action": "store_true",
                            "default": False,
                        },
                    },
                },
            ],
        },
        "steps": OrderedDict(),
    }

    cli_builder = CLIBuilder("test", pipeline_conf)

    parser = cli_builder.argparser

    assert parser is not None

