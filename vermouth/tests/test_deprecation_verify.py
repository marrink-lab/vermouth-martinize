
import pytest
import os
import pathlib
from vermouth.forcefield import ForceField

def test_variables_deprecation_warning(tmp_path, caplog):
    """
    Test that a deprecation warning is issued when the [variables] section is populated.
    """
    # Create a directory for the force field
    ff_dir = tmp_path / 'my_ff'
    ff_dir.mkdir()
    (ff_dir / 'my_ff.ff').write_text("""
    [ variables ]
    my_var 10
    """)
    
    # Initialize ForceField from directory
    ForceField(directory=str(ff_dir))
    
    assert 'Force field variables have been superseded by pipeline.yaml since vermouth 2.0.0' in caplog.text

def test_variables_no_deprecation_warning(tmp_path, caplog):
    """
    Test that no deprecation warning is issued when the [variables] section is empty.
    """
    # Create a directory for the force field
    ff_dir = tmp_path / 'my_ff_empty'
    ff_dir.mkdir()
    (ff_dir / 'my_ff_empty.ff').write_text("""
    [ moleculetype ]
    GLY 1
    """)
    
    # Initialize ForceField from directory
    ForceField(directory=str(ff_dir))
    
    assert 'Force field variables have been superseded by pipelines since vermouth 2.0.0' not in caplog.text

def test_variables_empty_section_no_warning(tmp_path, caplog):
    """
    Test that an empty [variables] section doesn't trigger the warning.
    """
    ff_dir = tmp_path / 'my_ff_empty_sec'
    ff_dir.mkdir()
    (ff_dir / 'my_ff_empty_sec.ff').write_text("""
    [ variables ]
    """)
    # Note: read_ff might fail if [variables] is empty but defined? 
    # No, it just calls _parse_variables for each line. 
    # If there are no lines after [variables], it won't populate self.variables.

    ForceField(directory=str(ff_dir))
    
    assert 'Force field variables have been superseded by pipelines since vermouth 2.0.0' not in caplog.text
