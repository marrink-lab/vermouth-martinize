import sys
from pathlib import Path
from collections import OrderedDict
sys.path.insert(0, str(Path(__file__).parent))
import jsonschema
import pytest
import vermouth
from vermouth.forcefield import ForceField
from vermouth.martinize2_pipeline import (
    TYPE_MAP,
    PipelineConfigBuilder,
    build_mini_parser,
)
from vermouth.pipeline import (
    CLIBuilder,
    PipelineBuilder,
    _options_used_in_condition,
    _validate_raw_step_names,
    compose_pipeline_file,
    eval_condition,
    find_pipeline_yaml,
    iter_cli_flags,
    load_pipeline_configs,
    load_yaml_file,
    merge_pipeline_mapping,
    rename_variables,
    select_include_fragment,
    validate_cli_options,
    validate_step_names,
)

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
                "equal": {
                    "variable": "ff",
                    "cli": "bondedtypes",
                }
            },
        ]
    }

    cli_refs, variable_refs = _options_used_in_condition(condition)

    assert cli_refs == {"bondedtypes", "elastic"}
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
        "cli": {
            "flags": {
                "elastic": {},
            },
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
        "cli": {"flags": {}},
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
        "cli": {
            "flags": {
                "elastic": {},
            },
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
        "cli": {
            "flags": {
                "elastic": {},
            },
        },
        "steps": OrderedDict([
            (
                "dummy",
                {
                    "args": {
                        "arg": {
                            "cli": "elastic",
                        }
                    }
                },
            ),
        ]),
    }

    validate_cli_options(pipeline_conf)

def test_validate_cli_options_unknown_condition_cli():
    """
    Test that validate_cli_options raises a KeyError
    when a condition references an undefined CLI option.
    """
    pipeline_conf = {
        "cli": {"flags": {}},
        "condition": {
            "equal": {
                "cli": "elastic",
                "value": True,
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
        "cli": {
            "exclusive_groups": [
            {
                "flags": {
                    "elastic": {},
                }
            }
            ],
        },
        "args": {
            "arg": {
                "cli": "elastic",
            }
        },
    }

    validate_cli_options(pipeline_conf)


@pytest.mark.parametrize("reference", [
    {"args": {"option": {"cli": "shared"}}},
    {"condition": {"equal": {"cli": "shared", "value": True}}},
])
def test_validate_cli_options_rejects_flags_defined_in_sibling_pipeline(reference):
    """CLI references cannot use flags defined in a sibling pipeline."""
    pipeline_conf = {
        "steps": OrderedDict([
            (
                "from",
                {
                    "steps": OrderedDict([
                        (
                            "consumer",
                            reference,
                        ),
                    ]),
                },
            ),
            (
                "to",
                {
                    "cli": {"flags": {"shared": {"default": "value"}}},
                    "args": {},
                },
            ),
        ]),
    }

    with pytest.raises(KeyError, match="shared"):
        validate_cli_options(pipeline_conf, path="martinize2")


def test_validate_cli_options_rejects_flags_defined_in_descendant():
    pipeline_conf = {
        "condition": {"equal": {"cli": "shared", "value": True}},
        "steps": {"child": {"cli": {"flags": {"shared": {}}}}},
    }
    with pytest.raises(KeyError, match="shared"):
        validate_cli_options(pipeline_conf)


def test_validate_cli_options_does_not_treat_source_metadata_as_a_flag():
    pipeline_conf = {
        "cli": {"flags": {"$source": "pipeline.yaml"}},
        "args": {"option": {"cli": "$source"}},
    }
    with pytest.raises(KeyError, match=r"\$source"):
        validate_cli_options(pipeline_conf)


@pytest.mark.parametrize("same_options", [True, False])
def test_validate_cli_options_duplicate_flags(same_options):
    pipeline_conf = {
        "steps": {
            "first": {
                "cli": {"flags": {"shared": {"default": 1}}},
                "args": {"option": {"cli": "shared"}},
            },
            "second": {
                "cli": {"flags": {"shared": {"default": 1 if same_options else 2}}},
                "args": {"option": {"cli": "shared"}},
            },
        },
    }
    if same_options:
        validate_cli_options(pipeline_conf)
    else:
        with pytest.raises(ValueError, match="shared.*different options"):
            validate_cli_options(pipeline_conf)


def test_build_mini_parser_defaults():
    """
    Test that the mini parser returns the default values
    when no command-line arguments are given.
    """
    parser = build_mini_parser()

    args = parser.parse_args([])

    assert args.from_ff == "charmm"
    assert args.to_ff == "martini3001"
    assert args.pipeline == []
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
            "-from", "source",
            "-ff", "target",
            "-pipeline", "water",
            "-pipeline-dir", "my_pipelines",
            "-ff-dir", "extra_ff",
            "-map-dir", "extra_maps",
            "-list-ff",
            "-list-blocks"
        ])

    assert args.from_ff == "source"
    assert args.to_ff == "target"
    assert args.pipeline == ["water"]
    assert args.pipeline_dir == [Path("my_pipelines")]
    assert args.extra_ff_dir == [Path("extra_ff")]
    assert args.extra_map_dir == [Path("extra_maps")]
    assert args.list_ff
    assert args.list_blocks


def test_martinize2_cli_groups_are_declared_in_pipeline_configuration():
    """Groups are defined at the narrowest pipeline scope that owns their flags."""
    config = PipelineConfigBuilder("charmm", "martini3001").build_config()
    source = config["martinize2"]["steps"]["from"]
    target = config["martinize2"]["steps"]["to"]

    assert source["steps"]["input"]["cli"]["groups"][0]["title"] == "Input files"
    assert (
        source["steps"]["input"]["steps"]["pdb_to_universal"]["cli"]["groups"][0]
        ["title"]
        == "Debugging options"
    )
    assert (
        source["steps"]["protein_description"]["cli"]["groups"][0]["title"]
        == "Protein description"
    )
    assert (
        source["steps"]["protein_description"]["cli"]["groups"][1]["title"]
        == "Secondary structure handling"
    )
    go_group = source["steps"]["go_system"]["cli"]["groups"][0]
    assert go_group["title"] == "Virtual site based GoMartini"
    assert [group["title"] for group in target["cli"]["groups"]] == [
        "Force field selection",
        "Output files",
        "Apply water bias.",
    ]
    secondary = target["steps"]["secondary_structure"]
    assert [group["title"] for group in secondary["cli"]["groups"]] == [
        "Virtual site based GoMartini",
        "Protein elastic network",
    ]
    assert (
        target["steps"]["vermouth.pipeline_processors.ApplyPosresWrapper"][
            "cli"
        ]["groups"][0]["title"]
        == "Position restraints"
    )
    assert set(go_group["flags"]) == set(secondary["cli"]["groups"][0]["flags"])

    parser = CLIBuilder(
        "martinize2", config["martinize2"], type_map=TYPE_MAP
    ).argparser
    titles = {group.title for group in parser._action_groups}
    assert titles == {
        "positional arguments",
        "options",
        "Force field selection",
        "Input files",
        "Debugging options",
        "Protein description",
        "Secondary structure handling",
        "Force field selection",
        "Output files",
        "Apply water bias.",
        "Position restraints",
        "Virtual site based GoMartini",
        "Protein elastic network",
    }, titles
    expected_group_flags = {
        "Secondary structure handling": {"extdih", "dssp", "ss", "collagen"},
        "Virtual site based GoMartini": {
            "go", "go_write_file", "go_low", "go_up", "go_res_dist",
            "go_backbone", "go_atomname", "go_eps",
        },
        "Protein elastic network": {
            "elastic", "rb_unit", "rb_force_constant", "rb_lower_bound",
            "rb_upper_bound", "res_min_dist", "rb_decay_factor",
            "rb_decay_shift", "rb_decay_power", "rb_minimum_force",
            "rb_selection",
        },
    }
    for title, expected_flags in expected_group_flags.items():
        groups = [group for group in parser._action_groups if group.title == title]
        assert len(groups) == 1
        assert {action.dest for action in groups[0]._group_actions} == expected_flags
    with pytest.raises(SystemExit):
        parser.parse_args(["-f", "input.pdb", "-dssp", "-ss", "C"])


def test_mini_parser_force_field_group_is_available_in_full_help():
    """Pre-parser and source-pipeline force-field groups remain distinct."""
    mini_parser = build_mini_parser()
    config = PipelineConfigBuilder("charmm", "martini3001").build_config()
    parser = CLIBuilder(
        "martinize2", config["martinize2"], type_map=TYPE_MAP
    )
    parser.build_argparser(
        parents=[mini_parser],
        added_flags={
            "from_ff", "to_ff", "extra_ff_dir", "extra_map_dir", "list_ff",
            "pipeline", "pipeline_dir", "verbosity", "maxwarn",
        },
    )

    help_text = parser.argparser.format_help()

    assert help_text.count("Force field selection:") == 1
    assert "-ff TO_FF" in help_text
    assert "-list-blocks" in help_text
    assert "-go-write-file [GO_WRITE_FILE]" in help_text


def test_build_mini_parser_defers_ambiguous_pipeline_flag():
    """
    Pipeline flags that prefix mini-parser options are left for the full CLI.
    """
    parser = build_mini_parser()

    args, remaining = parser.parse_known_args(["-f", "input.pdb"])

    assert args.from_ff == "charmm"
    assert remaining == ["-f", "input.pdb"]


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
    Test that find_pipeline_yaml finds a force-field pipeline.
    """
    result = find_pipeline_yaml("charmm", [])

    assert result == vermouth.DATA_PATH / "force_fields" / "charmm" / "pipeline.yaml"


@pytest.mark.parametrize(
    "force_field",
    [
        "martini3001",
        "martini22",
        "martini22p",
        "elnedyn21",
        "elnedyn22",
        "elnedyn22p",
    ],
)
def test_find_pipeline_yaml_force_field_targets(force_field):
    """Each shipped Martini and ELNEDYN force field provides a pipeline."""
    assert find_pipeline_yaml(force_field, []) == (
        vermouth.DATA_PATH / "force_fields" / force_field / "pipeline.yaml"
    )


def test_find_pipeline_yaml_force_field_directory(tmp_path):
    """A loaded force field resolves its pipeline from its own directory."""
    directory = tmp_path / "custom"
    directory.mkdir()
    pipeline = directory / "pipeline.yaml"
    pipeline.write_text("martinize2: {}", encoding="utf-8")
    force_field = ForceField(directory)

    assert find_pipeline_yaml(force_field, []) == pipeline


@pytest.mark.parametrize(
    "force_field, bond_type",
    [
        ("martini3001", 1),
        ("martini22", 6),
        ("martini22p", 6),
        ("elnedyn21", 1),
        ("elnedyn22", 1),
        ("elnedyn22p", 1),
    ],
)
def test_force_field_target_pipeline_includes_base_martini(
    force_field,
    bond_type,
):
    """Each target wrapper composes the shared Martini processing pipeline."""
    document = PipelineConfigBuilder("charmm", force_field).build_config()

    assert list(document["martinize2"]["steps"]) == ["from", "to"]
    assert "mappings" in document["martinize2"]["steps"]["to"]["variables"]
    elastic_args = document["martinize2"]["steps"]["to"]["steps"][
        "secondary_structure"
    ]["steps"]["elastic"]["args"]
    assert elastic_args["bond_type"]["value"] == bond_type
    assert elastic_args["bb_atomname"]["value"] == "BB"
    assert "force_field" not in elastic_args


def test_secondary_structure_pipeline_preserves_processing_order():
    root = PipelineConfigBuilder("charmm", "martini3001").build_config()["martinize2"]
    target_steps = root["steps"]["to"]["steps"]
    assert list(target_steps) == [
        "martinize",
        "vermouth.RTPPolisher",
        "vermouth.pipeline_processors.ApplyPosresWrapper",
        "secondary_structure",
        "vermouth.OutputLogger",
        "vermouth.gmx.topology.TopologyWriter",
        "vermouth.pdb.WritePDB",
    ]
    steps = target_steps["secondary_structure"]["steps"]
    assert list(steps) == [
        "vermouth.pipeline_processors.GoModelWrapper",
        "vermouth.pipeline_processors.MergeChainsWrapper",
        "vermouth.pipeline_processors.NameMolTypeWrapper",
        "merge_all_for_elastic",
        "elastic",
        "vermouth.pipeline_processors.VirtualSiteCreatorWrapper",
        "vermouth.pipeline_processors.ComputeWaterBiasWrapper",
        "vermouth.UnstashAttributes",
        "vermouth.SortMoleculeAtoms",
    ]
    assert steps["merge_all_for_elastic"]["processor"] == "vermouth.MergeAllMolecules"


@pytest.mark.parametrize(
    "elastic, go, rb_unit, merge_expected, elastic_expected",
    [
        (False, None, "all", False, False),
        (True, None, "all", True, True),
        (True, None, "molecule", False, True),
        (True, True, "all", False, False),
        (True, Path("contacts.txt"), "all", False, False),
    ],
)
def test_secondary_structure_elastic_conditions(
    elastic, go, rb_unit, merge_expected, elastic_expected,
):
    root = PipelineConfigBuilder("charmm", "martini3001").build_config()["martinize2"]
    steps = root["steps"]["to"]["steps"]["secondary_structure"]["steps"]
    cli_args = {"elastic": elastic, "go": go, "rb_unit": rb_unit}
    assert eval_condition(steps["merge_all_for_elastic"]["condition"], cli_args, {}) == merge_expected
    assert eval_condition(steps["elastic"]["condition"], cli_args, {}) == elastic_expected


@pytest.mark.parametrize(
    "options, expected",
    [
        ([], ["NameMolTypeWrapper", "SortMoleculeAtoms"]),
        (
            ["-elastic", "-eunit", "all"],
            ["NameMolTypeWrapper", "MergeAllMolecules", "ElasticWrapper", "SortMoleculeAtoms"],
        ),
        (["-go", "-elastic", "-eunit", "all"], ["GoModelWrapper"]),
    ],
)
def test_secondary_structure_pipeline_executes_enabled_steps_in_order(options, expected):
    root = PipelineConfigBuilder("charmm", "martini3001").build_config()["martinize2"]
    target = root["steps"]["to"]
    cli_args = CLIBuilder("target", target, type_map=TYPE_MAP).parse_cli_args(options)
    pipeline = PipelineBuilder(target).build_pipeline(cli_args, {"ff": None, "mappings": {}})
    secondary = next(
        processor for processor in pipeline.processors
        if getattr(processor, "name", None) == "secondary_structure"
    )
    executed = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        def run_system(self, system):
            executed.append(self.name)
            return system

    for node in secondary.ordered_nodes:
        processor = secondary.nodes[node]["processor"]
        name = processor.__name__ if isinstance(processor, type) else type(processor).__name__
        secondary.nodes[node]["processor"] = Recorder(name)

    system = object()
    assert secondary.run_system(system) is system
    assert executed == expected


@pytest.mark.parametrize("path", [
    "force_fields/charmm/pipeline.yaml",
    "force_fields/amber/pipeline.yaml",
    "force_fields/gromos/pipeline.yaml",
    "pipelines/base_martini.yaml",
    "force_fields/martini22/pipeline.yaml",
    "force_fields/martini22p/pipeline.yaml",
    "force_fields/martini3001/pipeline.yaml",
    "force_fields/elnedyn21/pipeline.yaml",
    "force_fields/elnedyn22/pipeline.yaml",
    "force_fields/elnedyn22p/pipeline.yaml",
])
def test_bundled_pipeline_defines_referenced_flags_locally(path):
    root = compose_pipeline_file(vermouth.DATA_PATH / path)["martinize2"]
    validate_cli_options(root, path="martinize2")


@pytest.mark.parametrize("force_field", ["elnedyn21", "elnedyn22", "elnedyn22p"])
def test_elnedyn_elastic_flag_override(force_field):
    config = PipelineConfigBuilder("charmm", force_field).build_config()
    martini = PipelineConfigBuilder("charmm", "martini3001").build_config()
    elnedyn_target = config["martinize2"]["steps"]["to"]
    martini_target = martini["martinize2"]["steps"]["to"]
    for elnedyn_scope, martini_scope in [
        (elnedyn_target, martini_target),
        (
            elnedyn_target["steps"]["secondary_structure"],
            martini_target["steps"]["secondary_structure"],
        ),
    ]:
        elnedyn_groups = elnedyn_scope["cli"]["groups"]
        martini_groups = martini_scope["cli"]["groups"]
        assert [group["title"] for group in elnedyn_groups] == [
            group["title"] for group in martini_groups
        ]
        for elnedyn_group, martini_group in zip(elnedyn_groups, martini_groups):
            assert {
                flag for flag, _ in iter_cli_flags({"cli": {"groups": [elnedyn_group]}})
            } == {
                flag for flag, _ in iter_cli_flags({"cli": {"groups": [martini_group]}})
            }
    parser = CLIBuilder("martinize2", config["martinize2"], type_map=TYPE_MAP)
    assert parser.parse_cli_args(["-f", "input.pdb"])["elastic"] is True
    assert parser.parse_cli_args(["-f", "input.pdb", "-noelastic"])["elastic"] is False
    group = next(
        group for group in parser.argparser._action_groups
        if group.title == "Protein elastic network"
    )
    assert {action.dest for action in group._group_actions} == {
        "elastic", "rb_unit", "rb_force_constant", "rb_lower_bound",
        "rb_upper_bound", "res_min_dist", "rb_decay_factor",
        "rb_decay_shift", "rb_decay_power", "rb_minimum_force", "rb_selection",
    }


@pytest.mark.parametrize("fragment", ["from", "to"])
def test_shared_go_group_is_available_in_each_fragment(fragment):
    root = PipelineConfigBuilder("charmm", "martini3001").build_config()["martinize2"]
    parser = CLIBuilder("fragment", root["steps"][fragment], type_map=TYPE_MAP)
    args = ["-go"]
    if fragment == "from":
        args.extend(["-f", "input.pdb"])
    assert parser.parse_cli_args(args)["go"] is True
    groups = [
        group for group in parser.argparser._action_groups
        if group.title == "Virtual site based GoMartini"
    ]
    assert len(groups) == 1
    assert {action.dest for action in groups[0]._group_actions} == {
        "go", "go_write_file", "go_low", "go_up", "go_res_dist",
        "go_backbone", "go_atomname", "go_eps",
    }


def test_source_and_target_share_go_options_without_cross_scope_references():
    config = PipelineConfigBuilder("charmm", "martini3001").build_config()
    root = config["martinize2"]
    source = root["steps"]["from"]
    target = root["steps"]["to"]
    validate_cli_options(source)
    validate_cli_options(target)

    parser = CLIBuilder("martinize2", root, type_map=TYPE_MAP)
    parsed = parser.parse_cli_args([
        "-f", "input.pdb", "-go", "-go-write-file", "-elastic", "-eunit", "all",
    ])
    assert parsed["go"] is True
    assert parsed["go_write_file"] == Path("contact_map_martinize.out")
    assert parsed["elastic"] is True
    assert parsed["rb_unit"] == "all"
    assert parser.parse_cli_args(["-f", "input.pdb", "-go", "contacts.txt"])[
        "go"
    ] == Path("contacts.txt")


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
    in cli.flags.
    """
    pipeline_conf = {
        "cli": {"flags": {"ff": {}, "go": {}}},
    }

    result = list(iter_cli_flags(pipeline_conf))

    assert result == [
        ("ff", {}),
        ("go", {}),
    ]

def test_iter_cli_flags_group():
    """
    Test that iter_cli_flags yields flags defined
    in cli.exclusive_groups.
    """
    pipeline_conf = {
        "cli": {
            "exclusive_groups": [
            {
                "flags": {
                    "elastic": {},
                }
            }
            ],
        },
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
        "steps": OrderedDict([
            (
                "dummy",
                {
                    "cli": {
                        "flags": {
                            "inpath": {},
                        },
                    }
                },
            ),
        ]),
    }

    result = list(iter_cli_flags(pipeline_conf))

    assert result == [
        ("inpath", {}),
    ]


def test_pipeline_config_builder_generates_source_target_document(tmp_path):
    """Source and target selections are resolved through generated includes."""
    (tmp_path / "source.yaml").write_text(
        """
martinize2:
  steps: !!omap
    - from:
        variables: [ff]
        steps: !!omap
          - read:
              args: {}
""",
        encoding="utf-8",
    )
    (tmp_path / "target.yaml").write_text(
        """
martinize2:
  steps: !!omap
    - to:
        steps: !!omap
          - write:
              args: {}
""",
        encoding="utf-8",
    )

    document = PipelineConfigBuilder(
        "source",
        "target",
        pipeline_dirs=[tmp_path],
    ).build_config()

    assert document["$schema"] == "./pipeline-schema.yaml"
    root = document["martinize2"]
    assert list(root["steps"]) == ["from", "to"]
    assert list(root["steps"]["from"]["steps"]) == ["read"]
    assert list(root["steps"]["to"]["steps"]) == ["write"]
    assert "$include" not in root["steps"]["from"]
    assert "$include" not in root["steps"]["to"]


@pytest.mark.parametrize("define_locally", [False, True])
def test_composed_source_cannot_borrow_target_flag(tmp_path, define_locally):
    source_cli = "        cli:\n          flags:\n            shared: {default: value}\n"
    (tmp_path / "source.yaml").write_text(
        "martinize2:\n"
        "  steps: !!omap\n"
        "    - from:\n"
        "        variables: [ff]\n"
        + (source_cli if define_locally else "")
        + "        steps: !!omap\n"
        "          - read:\n"
        "              args:\n"
        "                option: {cli: shared}\n",
        encoding="utf-8",
    )
    (tmp_path / "target.yaml").write_text(
        "martinize2:\n"
        "  steps: !!omap\n"
        "    - to:\n"
        "        cli:\n"
        "          flags:\n"
        "            shared: {default: value}\n"
        "        steps: !!omap\n"
        "          - write:\n"
        "              args: {}\n",
        encoding="utf-8",
    )
    builder = PipelineConfigBuilder("source", "target", pipeline_dirs=[tmp_path])
    if define_locally:
        builder.build_config()
    else:
        with pytest.raises(KeyError, match="shared"):
            builder.build_config()