import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pytest
from api import _cys_argument, water_bias, ignore_resname, maxwarn

def test_cys_argument_float():
    """
    Test that _cys_argument converts a numeric string
    to a float.
    """
    assert _cys_argument("1.5") == 1.5


def test_cys_argument_auto():
    """
    Test that _cys_argument accepts auto as a valid value.
    """
    assert _cys_argument("auto") == "auto"


def test_cys_argument_none():
    """
    Test that _cys_argument accepts none as a valid value.
    """
    assert _cys_argument("none") == "none"


def test_cys_argument_invalid():
    """
    Test that _cys_argument raises an ArgumentTypeError
    for an invalid value.
    """
    with pytest.raises(argparse.ArgumentTypeError):
        _cys_argument("wrong")


def test_water_bias_valid():
    """
    Test that water_bias splits a letter and epsilon value.
    """
    assert water_bias("H:3.6") == ("H", 3.6)


def test_water_bias_invalid():
    """
    Test that water_bias raises an ArgumentTypeError
    for invalid input.
    """
    with pytest.raises(argparse.ArgumentTypeError):
        water_bias("H")


def test_ignore_resname_single():
    """
    Test that ignore_resname returns a list with one residue name.
    """
    assert ignore_resname("HOH") == ["HOH"]


def test_ignore_resname_multiple():
    """
    Test that ignore_resname splits comma-separated residue names
    and removes surrounding whitespace.
    """
    assert ignore_resname("HOH, SOL, LIG") == ["HOH", "SOL", "LIG"]


def test_ignore_resname_empty_items():
    """
    Test that ignore_resname removes empty items.
    """
    assert ignore_resname("HOH,, SOL,") == ["HOH", "SOL"]

def test_maxwarn_count_only():
    """
    Test that maxwarn parses a count without
    a warning type.
    """
    assert maxwarn("3") == (None, 3)


def test_maxwarn_type_only():
    """
    Test that maxwarn parses a warning type
    without a count.
    """
    assert maxwarn("general") == ("general", None)


def test_maxwarn_type_and_count():
    """
    Test that maxwarn parses a warning type
    and count separated by a colon.
    """
    assert maxwarn("general:15") == ("general", 15)


def test_maxwarn_invalid():
    """
    Test that maxwarn raises an ArgumentTypeError
    for an invalid count.
    """
    with pytest.raises(argparse.ArgumentTypeError):
        maxwarn("general:abc")