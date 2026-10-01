from hypothesis import note, strategies as st
from hypothesis import given, settings 
from pathlib import Path
from collections import OrderedDict
import sys

import pytest 

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vermouth.pipeline import CLIBuilder


@st.composite
def cli_flags(draw, known_cli_flags, **opts):
    reserved_flags = ["help", "h"]
    cli_flags = st.dictionaries(
        keys=st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*", fullmatch=True).filter(lambda t: t not in known_cli_flags and t not in reserved_flags),
        values=st.fixed_dictionaries({}, optional={
            "default": st.none() | st.booleans() | st.integers() | st.text()
        }), **opts
    )
    flags = draw(cli_flags)
    known_cli_flags.update(flags)
    return flags


def cli_group(known_cli_flags, **opts):
    group = st.lists(st.fixed_dictionaries({
        "flags": cli_flags(known_cli_flags, min_size=2, max_size=4),
    }), **opts)
    return group


def collect_cli_definitions(config):
    """Collect flags and exclusive groups from a generated configuration."""
    flags = {}
    groups = []
    cli = config["cli"]
    flags.update(cli.get("flags", {}))
    groups.extend(cli.get("exclusive_groups", []))
    for step in config["steps"].values():
        step_flags, step_groups = collect_cli_definitions(step)
        flags.update(step_flags)
        groups.extend(step_groups)
    return flags, groups



@st.composite
def build_cli_conf(draw, known_cli_flags=None, *, min_depth=0, max_depth=3, depth=0):
    known_cli_flags = known_cli_flags if known_cli_flags is not None else {}
    new_cli_groups = []

    go_deeper = (draw(st.booleans()) and depth < max_depth) or depth < min_depth
    this_depth = st.fixed_dictionaries({}, optional={
        "flags": cli_flags(known_cli_flags, min_size=1, max_size=4),
        "exclusive_groups": cli_group(
            known_cli_flags,
            min_size=1,
            max_size=3,
        ),
    })
    this_depth = draw(this_depth)

    groups = this_depth.get('exclusive_groups', [])
    new_cli_groups.extend(groups)

    if go_deeper:
        confs = []
        for _ in range(draw(st.integers(min_value=1, max_value=3))):
            # This *must* be done in a loop, step-by-step, to keep known_cli_flags up to date along the way
            name, (conf, flags, groups) = draw(st.tuples(st.text(min_size=1), build_cli_conf(known_cli_flags, min_depth=min_depth, max_depth=max_depth, depth=depth+1)))
            confs.append((name, conf))
            new_cli_groups.extend(groups)
        rest = {"steps": OrderedDict(confs)}
    else:
        rest = {"steps": OrderedDict()}
    config = dict(cli=this_depth, **rest)
    flags, groups = collect_cli_definitions(config)
    return config, flags, groups



@settings(max_examples=500)
@given(st.data())
def test_something(data):
    conf, flags, groups = data.draw(build_cli_conf())
    cli_builder = CLIBuilder("test", conf)
    note(f'{groups=}')
    cli_args = []
    ungrouped_flags = set(flags)
    for group in groups:
        # Pick at most (exactly?) one arg per group
        cli_args.append(data.draw(st.sampled_from(sorted(group['flags']))))
        ungrouped_flags -= set(group['flags'])
    cli_args.extend(ungrouped_flags)
    note(f'{cli_args=}')
    args = []
    for flag in cli_args:
        args.extend([f'-{flag}', 'value'])

    try:
        cli_builder.build_argparser(exit_on_error=False)
        args = cli_builder.parse_cli_args(args)
    except Exception as e:
        pytest.fail(str(e))

    for flag in cli_args:
        assert args[flag] == "value"