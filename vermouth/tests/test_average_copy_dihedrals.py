# Copyright 2026 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test for the AverageCopyDihedrals processor.
"""
import pytest
from vermouth import System
from vermouth.molecule import Molecule
from vermouth.processors.average_copy_dihedrals import AverageCopyDihedrals

GROUP = "SC-BB-BB-SC scFix"


def make_molecule(angles, resnames=('ALA', 'GLY'), secstruct='CC', charge=0, chain='A'):
    """
    Two residues of BB and SC1 beads in `chain`, with one scFix dihedral per
    angle in `angles` and one other dihedral of 45 degrees, each on different
    atoms. `secstruct` gives the cgsecstruct of each residue and `charge` the
    charge of the last bead.
    """
    molecule = Molecule()
    residues = zip(resnames, secstruct)
    beads = ((resname, atomname, ss) for resname, ss in residues for atomname in ('BB', 'SC1'))
    for idx, (resname, atomname, ss) in enumerate(beads):
        molecule.add_node(idx, resname=resname, resid=idx // 2 + 1, atomname=atomname, chain=chain,
                          cgsecstruct=ss, charge=0)
    molecule.nodes[3]['charge'] = charge
    for angle, atoms in zip(angles, ([1, 0, 2, 3], [3, 2, 0, 1])):
        molecule.add_interaction('dihedrals', atoms, ['1', angle, '75', '1'],
                                 meta={'group': GROUP})
        molecule.add_interaction('dihedrals', atoms, ['1', '45.0', '75', '1'])
    return molecule


@pytest.mark.parametrize('angles_a, angles_b, expected', (
    # simple mean, keeping the number of decimals
    (['10.0'], ['20.0'], ['15.0']),
    # circular mean around +/-180 degrees
    (['170.0'], ['-170.0'], ['180.0']),
    (['-179.5'], ['179.5'], ['180.0']),
    # several dihedrals are averaged independently
    (['10.00', '-30.00'], ['30.00', '-10.00'], ['20.00', '-20.00']),
))
def test_average_copy_dihedrals(angles_a, angles_b, expected):
    """
    The scFix dihedrals of copies are averaged; other dihedrals are not.
    """
    copy_a = make_molecule(angles_a)
    copy_b = make_molecule(angles_b)
    system = System()
    system.molecules = [copy_a, copy_b]

    AverageCopyDihedrals().run_system(system)

    for molecule in (copy_a, copy_b):
        scfix = [d.parameters[1] for d in molecule.interactions['dihedrals']
                 if d.meta.get('group') == GROUP]
        other = [d.parameters[1] for d in molecule.interactions['dihedrals']
                 if d.meta.get('group') != GROUP]
        # -180 and 180 are the same angle
        assert [float(angle) % 360 for angle in scfix] == pytest.approx(
            [float(angle) % 360 for angle in expected])
        assert [len(angle.partition('.')[2]) for angle in scfix] == [
            len(angle.partition('.')[2]) for angle in expected]
        assert other == ['45.0'] * len(expected)
    assert copy_a.share_moltype_with(copy_b)


@pytest.mark.parametrize('kwargs_a, kwargs_b, averaged', (
    # modified in the same way, e.g. by -modify
    ({'charge': 1}, {'charge': 1}, True),
    # modified differently
    ({'charge': 1}, {}, False),
    # different sequence
    ({}, {'resnames': ('GLY', 'ALA')}, False),
    # same sequence, different secondary structure
    ({}, {'secstruct': 'CE'}, False),
))
def test_average_copy_dihedrals_copies(kwargs_a, kwargs_b, averaged):
    """
    Only molecules that share a moltype after the averaging are averaged,
    whatever made them the same.
    """
    molecule_a = make_molecule(['10.0'], **kwargs_a)
    molecule_b = make_molecule(['20.0'], **kwargs_b)
    system = System()
    system.molecules = [molecule_a, molecule_b]

    AverageCopyDihedrals().run_system(system)

    angles = [molecule.interactions['dihedrals'][0].parameters[1]
              for molecule in (molecule_a, molecule_b)]
    assert angles == (['15.0', '15.0'] if averaged else ['10.0', '20.0'])
    assert molecule_a.share_moltype_with(molecule_b) == averaged


@pytest.mark.parametrize('angles, expected', (
    # the mean changes both angles by 25 degrees
    (['10.0', '60.0'], None),
    # by 35 degrees
    (['10.0', '80.0'], 'by 35 degrees in chain A'),
    # by 35 degrees, across +/-180 degrees
    (['170.0', '-120.0'], 'by 35 degrees in chain A'),
    # the mean (36.6) changes the angle of chain C most
    (['10.0', '10.0', '100.0'], 'by 63 degrees in chain C'),
))
def test_average_copy_dihedrals_warning(caplog, angles, expected):
    """
    Warn if the averaging changes the angle of a copy by more than 30 degrees.
    """
    system = System()
    system.molecules = [make_molecule([angle], chain=chain) for angle, chain in zip(angles, 'ABC')]

    AverageCopyDihedrals().run_system(system)

    warnings = [record for record in caplog.records if record.levelname == 'WARNING']
    if expected is None:
        assert not warnings
    else:
        assert [record.type for record in warnings] == ['scfix-spread']
        assert f"scFix dihedral SC1-BB-BB-SC1 of ALA1-GLY2 {expected}." in caplog.text
