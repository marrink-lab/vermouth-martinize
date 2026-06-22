"""
Tests for the Pipeline Builder class.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pytest
from api import set_values, eval_condition, PipelineBuilder
from vermouth.processors.processor import Pipeline




def test_eval_condition_requires_one_key():
    """
    The condition must have exactly one key, which specifies the type of condition."""
    with pytest.raises(ValueError):
        eval_condition(
            {"equal": {"cli": "x", "value": 1}, "not": {}},
            {},
            {},
        )


def test_eval_condition_equal_cli_true():
    """
    Test that eval_condition returns True for an equal condition matching a CLI argument.
    """
    assert eval_condition(
        {"equal": {"cli": "x", "value": 1}},
        {"x": 1},
        {},
    ) is True


def test_eval_condition_equal_cli_false():
    """
    Test that eval_condition returns False for an equal condition not matching a CLI argument.
    """
    assert eval_condition(
        {"equal": {"cli": "x", "value": 1}},
        {"x": 2},
        {},
    ) is False


def test_eval_condition_equal_variable():
    """
    Test that eval_condition returns True for an equal condition matching a variable.
    """
    assert eval_condition(
        {"equal": {"variable": "ff", "value": "martini"}},
        {},
        {"ff": "martini"},
    ) is True


def test_eval_condition_equal_missing_cli_or_variable():
    """
    Test that eval_condition raises an error if the equal condition is missing the cli or variable key.
    """
    with pytest.raises(ValueError):
        eval_condition(
            {"equal": {"value": True}},
            {},
            {},
        )


def test_eval_condition_not():
    """
    Test that eval_condition returns the negation of the inner condition for a not condition.
    """
    assert eval_condition(
        {"not": {"equal": {"cli": "x", "value": True}}},
        {"x": False},
        {},
    ) is True


def test_eval_condition_all():
    """
    Test that eval_condition returns True for an all condition if all inner conditions are True.
    """
    assert eval_condition(
        {
            "all": [
                {"equal": {"cli": "a", "value": True}},
                {"equal": {"cli": "b", "value": True}},
            ]
        },
        {"a": True, "b": True},
        {},
    ) is True


def test_eval_condition_any():
    """
    Test that eval_condition returns True for an any condition if at least one inner condition is True.
    """
    assert eval_condition(
        {
            "any": [
                {"equal": {"cli": "a", "value": True}},
                {"equal": {"cli": "b", "value": True}},
            ]
        },
        {"a": False, "b": True},
        {},
    ) is True


def test_eval_condition_has_variable():
    """
    Test that eval_condition returns True for a has_variable condition if the specified variable exists.
    """
    class DummyForceField:
        variables = {"bondedtypes": "something"}

    assert eval_condition(
        {"has_variable": {"variable": "ff", "key": "bondedtypes"}},
        {},
        {"ff": DummyForceField()},
    ) is True


def test_eval_condition_unknown_type():
    """
    Test that eval_condition raises an error for an unknown condition type.
    """
    with pytest.raises(ValueError):
        eval_condition(
            {"unknown": {}},
            {},
            {},
        )
    
def test_set_values_fixed_value():
    """
    Test that set_values replaces a fixed value argument
    with the actual value.
    """
    pipeline_conf = {
        "args": {
            "delete_unknown": {
                "value": True,
            }
        }
    }

    cli_args = {}
    variables = {}

    set_values(pipeline_conf, cli_args, variables)

    assert pipeline_conf["args"]["delete_unknown"] is True


def test_set_values_cli_argument():
    """
    Test that set_values replaces a CLI argument reference
    with the value from the cli_args dictionary.
    """
    pipeline_conf = {
        "args": {
            "path": {
                "cli": "inpath",
            }
        }
    }

    cli_args = {
        "inpath": Path("test.pdb"),
    }

    variables = {}

    set_values(pipeline_conf, cli_args, variables)

    assert pipeline_conf["args"]["path"] == Path("test.pdb")

def test_set_values_invalid_argument_reference():
    """
    Test that set_values raises a KeyError when an argument
    has no value, cli, or variable reference.
    """
    pipeline_conf = {
        "args": {
            "path": {
                "wrong": "inpath",
            }
        }
    }

    with pytest.raises(KeyError):
        set_values(pipeline_conf, {}, {})

def test_set_values_recurses_into_steps_and_imports_processor():
    """
    Test that set_values recursively processes nested steps
    and imports the processor class for leaf steps.
    """
    pipeline_conf = {
        "steps": [
            (
                "pathlib.Path",
                {
                    "args": {
                        "path": {
                            "cli": "inpath",
                        }
                    }
                },
            )
        ]
    }

    cli_args = {
        "inpath": "test.pdb",
    }

    set_values(pipeline_conf, cli_args, {})

    step = pipeline_conf["steps"][0][1]

    assert step["condition"] is True
    assert step["args"]["path"] == "test.pdb"
    assert step["processor"] is Path

def test_set_values_condition():
    """
    Test that set_values evaluates and replaces a condition.
    """
    pipeline_conf = {
        "condition": {
            "equal": {
                "cli": "x",
                "value": True,
            }
        }
    }

    cli_args = {
        "x": True,
    }

    set_values(pipeline_conf, cli_args, {})

    assert pipeline_conf["condition"] is True

def test_set_values_variable_argument():
    """
    Test that set_values replaces a variable reference
    with the value from the variables dictionary.
    """
    force_field = object()

    pipeline_conf = {
        "args": {
            "ff": {
                "variable": "charmm.ff",
            }
        }
    }

    cli_args = {}

    variables = {
        "charmm.ff": force_field,
    }

    set_values(pipeline_conf, cli_args, variables)

    assert pipeline_conf["args"]["ff"] is force_field


def test_pipeline_builder_builds_pipeline():
    """
    Test that PipelineBuilder fills the pipeline configuration
    and builds a Pipeline object from it.
    """
    pipeline_conf = {
        "steps": [
            (
                "pathlib.Path",
                {
                    "args": {
                        "path": {
                            "cli": "inpath",
                        }
                    }
                },
            )
        ]
    }

    cli_args = {
        "inpath": "test.pdb",
    }

    variables = {}

    builder = PipelineBuilder(pipeline_conf)
    pipeline = builder.build_pipeline(cli_args, variables)

    assert isinstance(pipeline, Pipeline)
    assert pipeline_conf["steps"][0][1]["args"]["path"] == "test.pdb"
    assert pipeline_conf["steps"][0][1]["processor"] is Path



