import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pytest
from api import PipelineConfigBuilder, _options_used_in_condition, combine_pipeline_configs, find_pipeline_yaml, iter_cli_flags, load_pipeline_configs, load_yaml_file, validate_cli_options, build_mini_parser, namespace_variables

def test_options_used_in_condition_equal():
    """
    Test that an equal condition returns the CLI option
    used by the condition.
    """
    condition = {
        "equal": {
            "cli": "elastic",
            "value": True,
        }
    }

    cli_refs, variable_refs = _options_used_in_condition(condition)

    assert cli_refs == {"elastic"}
    assert variable_refs == set()


def test_options_used_in_condition_has_variable():
    """
    Test that a has_variable condition returns the variable
    used by the condition.
    """
    condition = {
        "has_variable": {
            "variable": "ff",
            "key": "bondedtypes",
        }
    }

    cli_refs, variable_refs = _options_used_in_condition(condition)

    assert cli_refs == set()
    assert variable_refs == {"ff"}


def test_options_used_in_condition_not():
    """
    Test that a not condition returns the references used
    by its nested condition.
    """
    condition = {
        "not": {
            "equal": {
                "cli": "go",
                "value": None,
            }
        }
    }

    cli_refs, variable_refs = _options_used_in_condition(condition)

    assert cli_refs == {"go"}
    assert variable_refs == set()


def test_options_used_in_condition_all():
    """
    Test that an all condition combines the references
    from all nested conditions.
    """
    condition = {
        "all": [
            {
                "equal": {
                    "cli": "elastic",
                    "value": True,
                }
            },
            {
                "has_variable": {
                    "variable": "ff",
                    "key": "bondedtypes",
                }
            },
        ]
    }

    cli_refs, variable_refs = _options_used_in_condition(condition)

    assert cli_refs == {"elastic"}
    assert variable_refs == {"ff"}


def test_options_used_in_condition_any():
    """
    Test that an any condition combines the references
    from all nested conditions.
    """
    condition = {
        "any": [
            {
                "equal": {
                    "cli": "go",
                    "value": True,
                }
            },
            {
                "equal": {
                    "cli": "elastic",
                    "value": True,
                }
            },
        ]
    }

    cli_refs, variable_refs = _options_used_in_condition(condition)

    assert cli_refs == {"go", "elastic"}
    assert variable_refs == set()


def test_options_used_in_condition_unknown_type():
    """
    Test that an unknown condition type raises a ValueError.
    """
    condition = {
        "unknown": {}
    }

    with pytest.raises(ValueError):
        _options_used_in_condition(condition)

def test_validate_cli_options_valid():
    """
    Test that validate_cli_options accepts a valid
    pipeline configuration.
    """
    pipeline_conf = {
        "cli_flags": {
            "elastic": {},
        },
        "args": {
            "arg": {
                "cli": "elastic",
            }
        },
    }

    validate_cli_options(pipeline_conf)

def test_validate_cli_options_unknown_cli():
    """
    Test that validate_cli_options raises a KeyError
    for an undefined CLI option.
    """
    pipeline_conf = {
        "cli_flags": {},
        "args": {
            "arg": {
                "cli": "elastic",
            }
        },
    }

    with pytest.raises(KeyError):
        validate_cli_options(pipeline_conf)

def test_validate_cli_options_unknown_variable():
    """
    Test that validate_cli_options raises a KeyError
    for an undefined variable.
    """
    pipeline_conf = {
        "variables": [],
        "args": {
            "arg": {
                "variable": "ff",
            }
        },
    }

    with pytest.raises(KeyError):
        validate_cli_options(pipeline_conf)

def test_validate_cli_options_condition():
    """
    Test that validate_cli_options accepts a condition
    that references a defined CLI option.
    """
    pipeline_conf = {
        "cli_flags": {
            "elastic": {},
        },
        "condition": {
            "equal": {
                "cli": "elastic",
                "value": True,
            }
        },
    }

    validate_cli_options(pipeline_conf)

def test_validate_cli_options_recursive_step():
    """
    Test that validate_cli_options recursively validates
    nested pipeline steps.
    """
    pipeline_conf = {
        "cli_flags": {
            "elastic": {},
        },
        "steps": [
            (
                "dummy",
                {
                    "args": {
                        "arg": {
                            "cli": "elastic",
                        }
                    }
                },
            )
        ],
    }

    validate_cli_options(pipeline_conf)

def test_validate_cli_options_unknown_condition_cli():
    """
    Test that validate_cli_options raises a KeyError
    when a condition references an undefined CLI option.
    """
    pipeline_conf = {
        "cli_flags": {},
        "condition": {
            "equal": {
                "cli": "elastic",
                "value": True,
            }
        },
    }

    with pytest.raises(KeyError):
        validate_cli_options(pipeline_conf)
    
def test_validate_cli_options_unknown_condition_variable():
    """
    Test that validate_cli_options raises a KeyError
    when a condition references an undefined variable.
    """
    pipeline_conf = {
        "variables": [],
        "condition": {
            "has_variable": {
                "variable": "ff",
                "key": "bondedtypes",
            }
        },
    }

    with pytest.raises(KeyError):
        validate_cli_options(pipeline_conf)

def test_validate_cli_options_cli_group():
    """
    Test that validate_cli_options accepts CLI options
    defined in a CLI group.
    """
    pipeline_conf = {
        "cli_groups": [
            {
                "flags": {
                    "elastic": {},
                }
            }
        ],
        "args": {
            "arg": {
                "cli": "elastic",
            }
        },
    }

    validate_cli_options(pipeline_conf)


def test_build_mini_parser_defaults():
    """
    Test that the mini parser returns the default values
    when no command-line arguments are given.
    """
    parser = build_mini_parser()

    args = parser.parse_args([])

    assert args.pipeline == ["charmm", "martini3001"]
    assert args.pipeline_dir == []
    assert args.extra_ff_dir == []
    assert args.extra_map_dir == []
    assert args.list_ff is False

def test_build_mini_parser_custom_arguments():
    """
    Test that the mini parser correctly parses custom
    command-line arguments.
    """
    parser = build_mini_parser()

    args = parser.parse_args([
        "-pipeline", "charmm", "water", "martini3001",
        "-pipeline-dir", "my_pipelines",
        "-extra_ff_dir", "extra_ff",
        "-extra_map_dir", "extra_maps",
        "-list_ff",
    ])

    assert args.pipeline == ["charmm", "water", "martini3001"]
    assert args.pipeline_dir == [Path("my_pipelines")]
    assert args.extra_ff_dir == [Path("extra_ff")]
    assert args.extra_map_dir == [Path("extra_maps")]
    assert args.list_ff is True

def test_namespace_variables_dict():
    """
    Test that namespace_variables namespaces a variable
    reference inside a dictionary.
    """
    obj = {
        "args": {
            "force_field": {
                "variable": "ff",
            }
        }
    }

    result = namespace_variables(obj, "charmm")

    assert result is obj
    assert obj["args"]["force_field"]["variable"] == "charmm.ff"

def test_namespace_variables_list():
    """
    Test that namespace_variables namespaces variable
    references inside a list.
    """
    obj = [
        {
            "variable": "ff",
        },
        {
            "value": True,
        },
    ]

    namespace_variables(obj, "martini3001")

    assert obj[0]["variable"] == "martini3001.ff"
    assert obj[1]["value"] is True

def test_namespace_variables_tuple():
    """
    Test that namespace_variables namespaces variable
    references inside a tuple.
    """
    obj = (
        {
            "variable": "ff",
        },
        {
            "variable": "mappings",
        },
    )

    namespace_variables(obj, "martini3001")

    assert obj[0]["variable"] == "martini3001.ff"
    assert obj[1]["variable"] == "martini3001.mappings"

def test_find_pipeline_yaml_full_path(tmp_path):
    """
    Test that find_pipeline_yaml returns a user-provided
    path when it exists.
    """
    file = tmp_path / "test.yaml"
    file.write_text("test")

    result = find_pipeline_yaml(str(file), [])

    assert result == file

def test_find_pipeline_yaml_pipeline_dir(tmp_path):
    """
    Test that find_pipeline_yaml finds a YAML file
    in a user-provided pipeline directory.
    """
    file = tmp_path / "charmm.yaml"
    file.write_text("test")

    result = find_pipeline_yaml("charmm", [tmp_path])

    assert result == file

def test_find_pipeline_yaml_default_directory():
    """
    Test that find_pipeline_yaml finds a YAML file
    in the default pipelines directory.
    """
    result = find_pipeline_yaml("charmm", [])

    assert result.name == "charmm.yaml"

def test_find_pipeline_yaml_not_found():
    """
    Test that find_pipeline_yaml raises a FileNotFoundError
    when the YAML file cannot be found.
    """
    with pytest.raises(FileNotFoundError):
        find_pipeline_yaml("this_file_does_not_exist", [])

def test_load_yaml_file(tmp_path):
    """
    Test that load_yaml_file loads a YAML file
    into a Python dictionary.
    """
    file = tmp_path / "test.yaml"
    file.write_text(
        """
        name: test
        number: 42
        """,
        encoding="utf-8",
    )

    result = load_yaml_file(file)

    assert result == {
        "name": "test",
        "number": 42,
    }

import pytest

def test_load_yaml_file_not_found():
    """
    Test that load_yaml_file raises a FileNotFoundError
    for a missing YAML file.
    """
    with pytest.raises(FileNotFoundError):
        load_yaml_file("this_file_does_not_exist.yaml")

def test_load_pipeline_configs_multiple(tmp_path):
    """
    Test that load_pipeline_configs loads multiple
    pipeline configurations.
    """
    (tmp_path / "charmm.yaml").write_text(
        "martinize2:\n  steps: []",
        encoding="utf-8",
    )

    (tmp_path / "water.yaml").write_text(
        "martinize2:\n  steps: []",
        encoding="utf-8",
    )

    configs = load_pipeline_configs(
        ["charmm", "water"],
        [tmp_path],
    )

    assert len(configs) == 2
    assert configs[0][0] == "charmm"
    assert configs[1][0] == "water"

def test_iter_cli_flags():
    """
    Test that iter_cli_flags yields flags defined
    in cli_flags.
    """
    pipeline_conf = {
        "cli_flags": {
            "ff": {},
            "go": {},
        }
    }

    result = list(iter_cli_flags(pipeline_conf))

    assert result == [
        ("ff", {}),
        ("go", {}),
    ]

def test_iter_cli_flags_group():
    """
    Test that iter_cli_flags yields flags defined
    in cli_groups.
    """
    pipeline_conf = {
        "cli_groups": [
            {
                "flags": {
                    "elastic": {},
                }
            }
        ]
    }

    result = list(iter_cli_flags(pipeline_conf))

    assert result == [
        ("elastic", {}),
    ]

def test_iter_cli_flags_recursive():
    """
    Test that iter_cli_flags yields flags
    from nested pipeline steps.
    """
    pipeline_conf = {
        "steps": [
            (
                "dummy",
                {
                    "cli_flags": {
                        "inpath": {},
                    }
                },
            )
        ]
    }

    result = list(iter_cli_flags(pipeline_conf))

    assert result == [
        ("inpath", {}),
    ]


def test_combine_pipeline_configs_combines_configs():
    """
    Test that combine_pipeline_configs combines variables,
    CLI flags, CLI groups, and steps from multiple configs.
    """
    configs = [
        (
            "charmm",
            {
                "martinize2": {
                    "variables": ["ff"],
                    "cli_flags": {
                        "inpath": {"type": "path"},
                    },
                    "cli_groups": [
                        {"flags": {"ss": {"type": "str"}}},
                    ],
                    "steps": [
                        ("ReadSystem", {"args": {}}),
                    ],
                }
            },
        ),
        (
            "martini3001",
            {
                "martinize2": {
                    "variables": ["ff", "mappings"],
                    "cli_flags": {
                        "outpath": {"type": "path"},
                    },
                    "steps": [
                        ("DoMapping", {"args": {}}),
                    ],
                }
            },
        ),
    ]

    combined = combine_pipeline_configs(configs)

    assert combined["variables"] == [
        "charmm.ff",
        "martini3001.ff",
        "martini3001.mappings",
    ]
    assert combined["cli_flags"] == {
        "inpath": {"type": "path"},
        "outpath": {"type": "path"},
    }
    assert combined["cli_groups"] == [
        {"flags": {"ss": {"type": "str"}}},
    ]
    assert combined["steps"] == [
        ("ReadSystem", {"args": {}}),
        ("DoMapping", {"args": {}}),
    ]

def test_combine_pipeline_configs_rejects_same_cli_flag_with_different_options():
    """
    Test that combine_pipeline_configs raises a ValueError
    when duplicate CLI flags have different option definitions.
    """
    configs = [
        (
            "first",
            {
                "martinize2": {
                    "cli_flags": {
                        "maxwarn": {"default": 0},
                    },
                }
            },
        ),
        (
            "second",
            {
                "martinize2": {
                    "cli_flags": {
                        "maxwarn": {"default": 1},
                    },
                }
            },
        ),
    ]

    with pytest.raises(ValueError):
        combine_pipeline_configs(configs)

def test_pipeline_config_builder_build_config(tmp_path):
    """
    Test that PipelineConfigBuilder loads, combines,
    validates, and returns pipeline configs.
    """
    file = tmp_path / "charmm.yaml"
    file.write_text(
        """
martinize2:
  cli_flags:
    inpath: {}
  steps: !!omap
    - ReadSystem:
        args:
          path:
            cli: inpath
""",
        encoding="utf-8",
    )

    builder = PipelineConfigBuilder(["charmm"], [tmp_path])

    configs, pipeline_conf = builder.build_config()

    assert configs[0][0] == "charmm"
    assert pipeline_conf["cli_flags"] == {"inpath": {}}
    assert pipeline_conf["steps"][0][0] == "ReadSystem"
    assert pipeline_conf["steps"][0][1]["args"]["path"]["cli"] == "inpath"