import sys
from pathlib import Path
from collections import OrderedDict
sys.path.insert(0, str(Path(__file__).parent))
import jsonschema
import pytest
import vermouth
from vermouth.pipeline import PipelineConfigBuilder, _options_used_in_condition, _validate_raw_step_names, combine_pipeline_configs, find_pipeline_yaml, iter_cli_flags, load_pipeline_configs, load_yaml_file, merge_pipeline_mapping, select_include_fragment, validate_cli_options, validate_step_names, build_mini_parser, rename_variables

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
            "--pipeline", "charmm", "water", "martini3001",
            "--pipeline-dir", "my_pipelines",
            "-extra_ff_dir", "extra_ff",
            "-extra_map_dir", "extra_maps",
            "-list_ff",
        ])

    assert args.pipeline == ["charmm", "water", "martini3001"]
    assert args.pipeline_dir == [Path("my_pipelines")]
    assert args.extra_ff_dir == [Path("extra_ff")]
    assert args.extra_map_dir == [Path("extra_maps")]
    assert args.list_ff is True

def test_rename_variables_updates_declarations_and_references():
    """
    Renaming applies to declarations and references in a nested mapping.
    """
    obj = {
        "variables": ["ff", "mappings"],
        "args": {"force_field": {"variable": "ff"}},
    }

    result = rename_variables(obj, {"ff": "source_ff"})

    assert result is obj
    assert obj["variables"] == ["source_ff", "mappings"]
    assert obj["args"]["force_field"]["variable"] == "source_ff"


def test_rename_variables_rewrites_collections():
    """
    Renaming applies recursively inside mutable and immutable collections.
    """
    obj = [
        {
            "variable": "ff",
        },
        {
            "value": True,
        },
    ]

    rename_variables(obj, {"ff": "source_ff"})

    assert obj[0]["variable"] == "source_ff"
    assert obj[1]["value"] is True


def test_rename_variables_rejects_unknown_source_variable():
    """Variable renames reject misspelled fragment-local names."""
    with pytest.raises(KeyError, match="undefined variable"):
        rename_variables({"variables": ["ff"]}, {"mappings": "source_mappings"})

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


def test_load_yaml_file_escapes_leading_dollar_keys_and_values(tmp_path):
    """
    A double leading dollar sign creates literal mapping keys and values.
    """
    file = tmp_path / "escaped-dollars.yaml"
    file.write_text(
        """
$$literal_key: $$literal_value
""",
        encoding="utf-8",
    )

    result = load_yaml_file(file)

    assert list(result) == ["$literal_key"]
    assert result["$literal_key"] == "$literal_value"


def test_escaped_dollars_are_not_composition_directives(tmp_path):
    """
    Escaped strategy keys and remove values are merged as literal strings.
    """
    file = tmp_path / "escaped-directives.yaml"
    file.write_text(
        """
$$strategy: literal_key
literal_value: $$remove
""",
        encoding="utf-8",
    )
    incoming = load_yaml_file(file)
    target = {"literal_value": "original"}

    merge_pipeline_mapping(target, incoming)

    assert target == {
        "$strategy": "literal_key",
        "literal_value": "$remove",
    }


def test_load_yaml_file_rejects_escaped_key_collisions(tmp_path):
    """
    Literal escaped keys cannot collide with directive keys after unescaping.
    """
    file = tmp_path / "escaped-key-collision.yaml"
    file.write_text(
        """
$strategy: merge
$$strategy: literal_key
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Escaped mapping key"):
        load_yaml_file(file)


def test_merge_pipeline_mapping_rejects_escaped_key_collisions(tmp_path):
    """
    Literal escaped keys cannot collide with keys from an earlier fragment.
    """
    file = tmp_path / "escaped-key.yaml"
    file.write_text(
        """
$$strategy: literal_key
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Escaped mapping key"):
        merge_pipeline_mapping({"$strategy": "merge"}, load_yaml_file(file))


def test_merge_pipeline_mapping_merges_by_default():
    """
    The default strategy recursively merges mappings, including arg sources.
    """
    target = {
        "args": {
            "source": {
                "cli": "input",
                "default": "fallback",
            }
        }
    }

    merge_pipeline_mapping(
        target,
        {
            "args": {
                "source": {
                    "value": "configured",
                }
            }
        },
    )

    assert target == {
        "args": {
            "source": {
                "cli": "input",
                "default": "fallback",
                "value": "configured",
            }
        }
    }


def test_merge_pipeline_mapping_replace_clears_the_target_mapping():
    """
    The replace strategy removes existing keys before applying an incoming map.
    """
    target = {
        "obsolete": True,
        "nested": {"previous": True},
    }

    merge_pipeline_mapping(
        target,
        {
            "$strategy": "replace",
            "replacement": True,
        },
    )

    assert target == {"replacement": True}


def test_merge_pipeline_mapping_remove_is_idempotent():
    """
    Removing an absent key succeeds without changing other values.
    """
    target = {"present": True}

    merge_pipeline_mapping(
        target,
        {
            "present": "$remove",
            "absent": "$remove",
        },
    )

    assert target == {}


def test_validate_step_names_rejects_duplicate_keys():
    """
    Step keys are unique within a pipeline, including nested pipelines.
    """
    pipeline_conf = {
        "steps": [
            ("mapping", {"processor": "pathlib.Path"}),
            ("mapping", {"processor": "pathlib.PurePath"}),
        ]
    }

    with pytest.raises(
        ValueError,
        match=(
            "martinize2.steps\\[0\\].mapping and "
            "martinize2.steps\\[1\\].mapping"
        ),
    ):
        _validate_raw_step_names(pipeline_conf)


def test_validate_step_names_reports_nested_collision_paths():
    """
    Duplicate step key errors identify both full nested step paths.
    """
    pipeline_conf = {
        "steps": [
            (
                "prepare",
                {
                    "steps": [
                        ("mapping", {"processor": "pathlib.Path"}),
                        ("mapping", {"processor": "pathlib.PurePath"}),
                    ]
                },
            )
        ]
    }

    with pytest.raises(
        ValueError,
        match=(
            "martinize2.steps\\[0\\].prepare.steps\\[0\\].mapping and "
            "martinize2.steps\\[0\\].prepare.steps\\[1\\].mapping"
        ),
    ):
        _validate_raw_step_names(pipeline_conf)


def test_validate_step_names_reports_source_file():
    """Duplicate step key errors identify the YAML source file."""
    pipeline_conf = {
        "steps": [
            ("mapping", {"processor": "pathlib.Path"}),
            ("mapping", {"processor": "pathlib.PurePath"}),
        ]
    }

    with pytest.raises(
        ValueError,
        match=r"martinize2\.steps\[1\]\.mapping in included\.yaml",
    ):
        _validate_raw_step_names(pipeline_conf, source="included.yaml")


def test_validate_step_names_allows_reused_processors():
    """
    Multiple named steps may use the same processor implementation.
    """
    pipeline_conf = {
        "steps": OrderedDict([
            ("first_path", {"processor": "pathlib.Path"}),
            ("second_path", {"processor": "pathlib.Path"}),
        ])
    }

    validate_step_names(pipeline_conf)


def test_merge_pipeline_mapping_inserts_between_paired_local_anchors():
    """
    Inserted steps use their mapping key rather than an id attribute.
    """
    steps = OrderedDict([
        ("first", {"processor": "pathlib.Path"}),
        ("last", {"processor": "pathlib.PurePath"}),
    ])

    merge_pipeline_mapping(
        steps,
        {
            "inserted": {
                "processor": "pathlib.Path",
                "$insert_after": "first",
                "$insert_before": "last",
            },
        },
    )

    assert list(steps) == [
        "first",
        "inserted",
        "last",
    ]
    assert steps["inserted"] == {"processor": "pathlib.Path"}


def test_merge_pipeline_mapping_merges_existing_step_at_anchored_position():
    """Existing step names merge when their local anchors are satisfied."""
    steps = OrderedDict([
        ("first", {"processor": "pathlib.Path"}),
        ("existing", {"processor": "pathlib.PurePath", "args": {}}),
        ("last", {"processor": "pathlib.Path"}),
    ])

    merge_pipeline_mapping(
        steps,
        {
            "existing": {
                "$insert_after": "first",
                "$insert_before": "last",
                "args": {"path": {"value": "configured"}},
            },
        },
        source="derived.yaml:martinize2.steps.existing",
    )

    assert list(steps) == ["first", "existing", "last"]
    assert steps["existing"] == {
        "processor": "pathlib.PurePath",
        "args": {"path": {"value": "configured"}},
    }


def test_merge_pipeline_mapping_rejects_misplaced_existing_step():
    """Existing step names fail when their requested local anchor is unmet."""
    steps = OrderedDict([
        ("first", {"processor": "pathlib.Path"}),
        ("existing", {"processor": "pathlib.PurePath"}),
        ("last", {"processor": "pathlib.Path"}),
    ])

    with pytest.raises(
        ValueError,
        match=(
            r"Cannot position 'existing' from derived\.yaml:"
            r"martinize2\.steps\.existing: it is not immediately after "
            r"local anchor 'last'"
        ),
    ):
        merge_pipeline_mapping(
            steps,
            {
                "existing": {
                    "$insert_after": "last",
                },
            },
            source="derived.yaml:martinize2.steps.existing",
        )


def test_merge_pipeline_mapping_reports_source_for_missing_local_anchor():
    """Missing insertion anchors identify the YAML fragment and step."""
    with pytest.raises(
        KeyError,
        match=(
            r"Cannot insert 'inserted' from included\.yaml:"
            r"martinize2\.steps\.inserted: local anchor 'missing' was not found"
        ),
    ):
        merge_pipeline_mapping(
            OrderedDict(),
            {
                "inserted": {
                    "processor": "pathlib.Path",
                    "$insert_after": "missing",
                },
            },
            source="included.yaml:martinize2.steps.inserted",
        )


def test_pipeline_schema_rejects_legacy_id():
    """
    The pipeline schema rejects the superseded id attribute.
    """
    schema = load_yaml_file(vermouth.DATA_PATH / "pipelines" / "pipeline-schema.yaml")
    config = {
        "martinize2": {
            "steps": [
                ["mapping", {"id": "legacy_mapping"}],
            ],
        },
    }

    with pytest.raises(jsonschema.ValidationError, match="'id'"):
        jsonschema.validate(config, schema)


def test_pipeline_schema_accepts_include_without_steps():
    """
    An include-only pipeline fragment is valid pipeline YAML.
    """
    schema = load_yaml_file(vermouth.DATA_PATH / "pipelines" / "pipeline-schema.yaml")
    config = {
        "martinize2": {
            "$include": ["common"],
        },
    }

    jsonschema.validate(config, schema)


def test_pipeline_schema_accepts_include_in_processor_args():
    """
    Includes can be declared inside processor argument mappings.
    """
    schema = load_yaml_file(vermouth.DATA_PATH / "pipelines" / "pipeline-schema.yaml")
    config = {
        "martinize2": {
            "steps": [
                [
                    "mapping",
                    {
                        "args": {
                            "$include": ["common.yaml:martinize2.steps[0].args"],
                        }
                    },
                ]
            ],
        },
    }

    jsonschema.validate(config, schema)


def test_pipeline_schema_accepts_include_variable_renames():
    """An include mapping may explicitly rename fragment-local variables."""
    schema = load_yaml_file(vermouth.DATA_PATH / "pipelines" / "pipeline-schema.yaml")
    config = {
        "martinize2": {
            "$include": [
                {
                    "path": "common.yaml:martinize2.steps.read_input",
                    "rename_variables": {"ff": "source_ff"},
                },
            ],
        },
    }

    jsonschema.validate(config, schema)


def test_include_renames_fragment_variable_references(tmp_path):
    """Variable renames apply only to the included fragment."""
    (tmp_path / "fragment.yaml").write_text(
        """
martinize2:
  steps: !!omap
    - read_input:
        args:
          force_field:
            variable: ff
""",
        encoding="utf-8",
    )
    including_path = tmp_path / "including.yaml"
    including_path.write_text(
        """
martinize2:
  variables:
    - source_ff
  steps: !!omap
    - consumer:
        args:
          $include:
            - path: fragment.yaml:martinize2.steps.read_input.args
              rename_variables:
                ff: source_ff
""",
        encoding="utf-8",
    )

    _, config = load_pipeline_configs([including_path])[0]

    assert (
        config["martinize2"]["steps"]["consumer"]["args"]["force_field"]["variable"]
        == "source_ff"
    )


def test_select_include_fragment_by_ordered_map_index_or_name(tmp_path):
    """
    Fragment selectors support both numeric and named ordered-map indexes.
    """
    source = tmp_path / "source.yaml"
    source.write_text(
        """
martinize2:
  steps: !!omap
    - read_input:
        args:
          path:
            value: input.pdb
""",
        encoding="utf-8",
    )
    config = load_yaml_file(source)

    by_index = select_include_fragment(config, "martinize2.steps[0].args")
    by_name = select_include_fragment(
        config,
        "martinize2.steps.read_input.args",
    )

    assert by_index == by_name == {
        "path": {
            "value": "input.pdb",
        }
    }


def test_load_pipeline_configs_rejects_duplicate_step_keys(tmp_path):
    """
    Duplicate keys from an ordered YAML mapping are rejected on load.
    """
    pipeline_file = tmp_path / "duplicate-steps.yaml"
    pipeline_file.write_text(
        """
martinize2:
  steps: !!omap
    - mapping:
        processor: pathlib.Path
    - mapping:
        processor: pathlib.PurePath
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate step key 'mapping'"):
        load_pipeline_configs([pipeline_file])

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


def test_load_pipeline_configs_resolves_includes(tmp_path):
    """
    Include directives are resolved in loaded configurations.
    """
    (tmp_path / "included.yaml").write_text(
        """
martinize2:
  steps: !!omap
    - included_step:
        processor: pathlib.Path
""",
        encoding="utf-8",
    )
    including_path = tmp_path / "including.yaml"
    including_path.write_text(
        """
martinize2:
  $include:
    - included.yaml:martinize2
  steps: !!omap
    - including_step:
        processor: pathlib.PurePath
""",
        encoding="utf-8",
    )

    configs = load_pipeline_configs([including_path])

    assert [name for name, _ in configs] == ["including"]
    root = configs[0][1]["martinize2"]
    assert "$include" not in root
    assert list(root["steps"]) == ["included_step", "including_step"]
    assert root["steps"]["included_step"]["$source"].endswith(
        "included.yaml:martinize2.steps.included_step"
    )
    assert root["steps"]["including_step"]["$source"].endswith(
        "including.yaml:martinize2.steps.including_step"
    )


def test_load_pipeline_configs_converts_validated_steps_to_ordered_dicts(tmp_path):
    """
    Validated ordered YAML mappings become OrderedDict step mappings.
    """
    pipeline_file = tmp_path / "ordered-steps.yaml"
    pipeline_file.write_text(
        """
martinize2:
  steps: !!omap
    - prepare:
        steps: !!omap
          - map:
              processor: pathlib.Path
""",
        encoding="utf-8",
    )

    _, config = load_pipeline_configs([pipeline_file])[0]
    steps = config["martinize2"]["steps"]

    assert isinstance(steps, OrderedDict)
    assert list(steps) == ["prepare"]
    assert isinstance(steps["prepare"]["steps"], OrderedDict)
    assert list(steps["prepare"]["steps"]) == ["map"]


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


def test_combine_pipeline_configs_rejects_duplicate_step_keys():
    """
    Combined fragments cannot introduce duplicate root step keys.
    """
    configs = [
        (
            "first",
            {
                "martinize2": {
                    "steps": OrderedDict([
                        ("mapping", {"processor": "pathlib.Path"}),
                    ]),
                }
            },
        ),
        (
            "second",
            {
                "martinize2": {
                    "steps": OrderedDict([
                        ("mapping", {"processor": "pathlib.PurePath"}),
                    ]),
                }
            },
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate step key 'mapping'"):
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