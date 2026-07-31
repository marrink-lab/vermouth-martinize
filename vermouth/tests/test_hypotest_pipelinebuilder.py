from hypothesis import note, strategies as st
from hypothesis import given, settings, note
from pathlib import Path
import sys
import copy 
import pytest 
import types

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vermouth.pipeline import PipelineBuilder
from vermouth.processors.processor import Pipeline, Processor


class DummyProcessor(Processor):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run_system(self, system):
        return system


dummy_module = types.ModuleType("dummy_processors")
dummy_module.DummyProcessor = DummyProcessor
sys.modules["dummy_processors"] = dummy_module

simple_value = st.none() | st.booleans() | st.integers() | st.text(max_size=10)


@st.composite
def pipeline_leaf(draw):
    cli_name = draw(st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*", fullmatch=True))
    variable_name = draw(st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*", fullmatch=True))

    cli_value = draw(simple_value)
    variable_value = draw(simple_value)
    fixed_value = draw(simple_value)

    conf = {
        "cli_flags": {
            cli_name: {"default": cli_value}
        },
        "variables": [
            variable_name
        ],
        "args": {
            "from_cli": {"cli": cli_name},
            "from_variable": {"variable": variable_name},
            "fixed": {"value": fixed_value},
        },
    }

    cli_args = {
        cli_name: cli_value,
    }

    variables = {
        variable_name: variable_value,
    }

    return conf, cli_args, variables


@st.composite
def build_pipeline_conf(draw, min_depth=0, max_depth=3, depth=0):
    go_deeper = (draw(st.booleans()) and depth < max_depth) or depth < min_depth

    if not go_deeper:
        return draw(pipeline_leaf())

    steps = []
    cli_args = {}
    variables = {}

    for _ in range(draw(st.integers(min_value=1, max_value=3))):
        MODULE_NAME = Path(__file__).stem
        step_name = "dummy_processors.DummyProcessor"

        step_conf, step_cli_args, step_variables = draw(
            build_pipeline_conf(
                min_depth=min_depth,
                max_depth=max_depth,
                depth=depth + 1,
            )
        )

        steps.append((step_name, step_conf))
        cli_args.update(step_cli_args)
        variables.update(step_variables)

    conf = {
        "steps": steps
    }

    return conf, cli_args, variables


@settings(max_examples=500)
@given(build_pipeline_conf(min_depth=1))
def test_pipeline_builder_builds_pipeline(conf):
    conf, cli_args, variables = conf

    note(f"{conf=}")
    note(f"{cli_args=}")
    note(f"{variables=}")

    builder = PipelineBuilder(copy.deepcopy(conf))

    try:
        pipeline = builder.build_pipeline(cli_args, variables)
    except Exception as e:
        pytest.fail(str(e))

    assert isinstance(pipeline, Pipeline)