# Copyright 2018 University of Groningen
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
Test that force field files are properly read.
"""

import textwrap
import pytest
import numpy as np
import vermouth.ffinput
import vermouth.forcefield
import vermouth.molecule
import vermouth.gmx.itp_read

class TestBlock:
    @staticmethod
    def test_moleculetype():
        lines = """
        [ moleculetype ]
        GLY  3
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        assert 'GLY' in ff.blocks
        assert ff.blocks['GLY'].nrexcl == 3

    @staticmethod
    def test_multiple_moleculetype():
        lines = """
            [ moleculetype ]
            GLY  3

            [ moleculetype ]
            VAL 2
            """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        assert set(ff.blocks) == {'GLY', 'VAL'}
        assert ff.blocks['GLY'].nrexcl == 3
        assert ff.blocks['VAL'].nrexcl == 2

    @staticmethod
    def test_atoms():
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
          1 P4 1 ALA BB 1
        2 P3\t1\tALA   SC1 2 -3 7
        3 P2 1 ALA SC2 3 {"custom": 4, "other": "plop"}
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        block = ff.blocks['GLY']
        assert len(block.nodes) == 3
        assert block.nodes['BB'] == {'atomname': 'BB', 'atype': 'P4',
                                     'resname': 'ALA', 'resid': 1,
                                     'charge_group': 1}
        assert block.nodes['SC1'] == {'atomname': 'SC1', 'atype': 'P3',
                                      'resname': 'ALA', 'resid': 1,
                                      'charge_group': 2, 'charge': -3.0,
                                      'mass': 7.0}
        assert block.nodes['SC2'] == {'atomname': 'SC2', 'atype': 'P2',
                                      'resname': 'ALA', 'resid': 1,
                                      'charge_group': 3, 'custom': 4,
                                      'other': 'plop'}

    @staticmethod
    def test_fixed_number_interaction():
        """
        Define an interaction for which the number of atoms required is known.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P3 1 ALA SC1 2
        3 P2 1 ALA SC2 3

        [ bonds ]
        1 2 1 0.2 100
        2 3 4 0.6 700 {"comment": "custom"}
        3 1 -- 9 plop toto
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        block = ff.blocks['GLY']

        bonds = [
            vermouth.molecule.Interaction(
                atoms=['BB', 'SC1'], parameters=['1', '0.2', '100'], meta={},
            ),
            vermouth.molecule.Interaction(
                atoms=['SC1', 'SC2'], parameters=['4', '0.6', '700'],
                meta={'comment': 'custom'},
            ),
            vermouth.molecule.Interaction(
                atoms=['SC2', 'BB'], parameters=['9', 'plop', 'toto'], meta={},
            ),
        ]
        assert block.interactions['bonds'] == bonds


    @staticmethod
   #def test_variable_number_interaction():
   #    """
   #    Define an interaction for which the number of atoms required is unknown.
   #    """
   #    lines = """
   #    [ moleculetype ]
   #    GLY 1

   #    [ atoms ]
   #    1 P4 1 ALA BB 1
   #    2 P3 1 ALA SC1 2
   #    3 P2 1 ALA SC2 3

   #    [ unknown ]
   #    1 2 -- 1 0.2 100
   #    2 3 1 -- 0.6 700 {"comment": "custom"}
   #    3 1 2
   #    """
   #    lines = textwrap.dedent(lines).splitlines()
   #    ff = vermouth.forcefield.ForceField(name='test_ff')
   #    vermouth.ffinput.read_ff(lines, ff)
   #    block = ff.blocks['GLY']

   #    interactions = [
   #        vermouth.molecule.Interaction(
   #            atoms=['BB', 'SC1'], parameters=['1', '0.2', '100'], meta={},
   #        ),
   #        vermouth.molecule.Interaction(
   #            atoms=['SC1', 'SC2', 'BB'], parameters=['0.6', '700'],
   #            meta={'comment': 'custom'},
   #        ),
   #        vermouth.molecule.Interaction(
   #            atoms=['SC2', 'BB', 'SC1'], parameters=[], meta={},
   #        ),
   #    ]
   #    assert block.interactions['unknown'] == interactions

    @staticmethod
    def test_interaction_by_name():
        """
        Create an interaction and refer to atoms by name.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P3 1 ALA SC1 2
        3 P2 1 ALA SC2 3

        [ bonds ]
        BB SC1 1 0.2 100
        SC1 SC2 4 0.6 700 {"comment": "custom"}
        SC2 BB -- 9 plop toto
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        block = ff.blocks['GLY']

        bonds = [
            vermouth.molecule.Interaction(
                atoms=['BB', 'SC1'], parameters=['1', '0.2', '100'], meta={},
            ),
            vermouth.molecule.Interaction(
                atoms=['SC1', 'SC2'], parameters=['4', '0.6', '700'],
                meta={'comment': 'custom'},
            ),
            vermouth.molecule.Interaction(
                atoms=['SC2', 'BB'], parameters=['9', 'plop', 'toto'], meta={},
            ),
        ]
        assert block.interactions['bonds'] == bonds

    # test_interaction_edges(interaction_lines, edges)
    @staticmethod
    @pytest.mark.parametrize(
        'interaction_lines, edges', (
            (  # regular bonds create edges
                """
                [ bonds ]
                BB SC1
                SC2 SC3
                """,
                (('BB', 'SC1'), ('SC2', 'SC3')),
            ),
            (  # regular angles create edges
                """
                [ angles ]
                BB SC1 SC2
                SC1 SC2 SC3
                """,
                (('BB', 'SC1'), ('SC1', 'SC2'), ('SC2', 'SC3')),
            ),
            (  # regular dihedrals create edges
                """
                [dihedrals]
                BB SC1 SC2 SC3
                """,
                (('BB', 'SC1'), ('SC1', 'SC2'), ('SC2', 'SC3')),
            ),
            (  # impropers do not create edges
                """
                [ impropers ]
                BB SC1 SC2 SC3
                """,
                (),
            ),
            (
                """
                [ dihedrals ]
                BB SC1 SC2 SC3 2 ; gromacs dihedrals of type 2 are impropers
                """,
                (),
            ),
            (  # the edge section creates edges
                """
                [ edges ]
                SC1 SC2
                BB SC3
                """,
                (('SC1', 'SC2'), ('BB', 'SC3')),
            ),
            (  # edges can be deactivated from interactions
                """
                [ bonds ]
                BB SC1 -- {"edge": false}
                SC2 SC3 -- {"edge": true}
                """,
                (('SC2', 'SC3'), ),
            ),
        )
    )
    def test_interaction_edges(interaction_lines, edges):
        """
        Edges are created where expected.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P3 1 ALA SC1 2
        3 P2 1 ALA SC2 3
        4 P2 1 ALA SC3 3
        """
        lines = textwrap.dedent(lines) + textwrap.dedent(interaction_lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        block = ff.blocks['GLY']

        assert (set(frozenset(edge) for edge in block.edges)
                == set(frozenset(edge) for edge in edges))

    # test_interaction_fail_reference
    @staticmethod
    @pytest.mark.parametrize('interaction_lines', (
        # Refers, by index, to an atom out of range
        """
        [ bonds ]
        1 6
        """,
        # Refers, by name, to a non-defined atom (SC2 does not exist)
        """
        [ bonds ]
        BB SC2
        """,
        # One missing atom in the definition (refers to only one atom instead of 2)
        """
        [ bonds ]
        BB
        """,
        # Prefixed atom in a block interaction
        """
        [ bonds ]
        BB +SC1
        """,
        """
        [ bonds ]
        BB -SC1
        """,
        """
        [ bonds ]
        BB <SC1
        """,
        """
        [ bonds ]
        BB >SC1
        """,
    ))
    def test_interaction_fail_reference(interaction_lines):
        """
        Faulty atom references in interactions lead to an exception.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        ; The following atoms have names that should not be valid
        2 P4 2 TRP +SC1 1
        3 P4 2 TRP -SC1 1
        4 P4 2 TRP <SC1 1
        5 P4 2 TRP >SC1 1
        """
        lines = textwrap.dedent(lines)
        lines = lines.splitlines()
        lines += textwrap.dedent(interaction_lines).splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
            vermouth.ffinput.read_ff(lines, ff)

    @staticmethod
    def test_interaction_hash_meta():
        """
        Make sure that the `#meta` lines are accounted for.
        """
        lines = """
        [ moleculetype ]
        GLY  3

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1
        3 P4 1 ALA SC2 1
        4 P4 1 ALA SC3 1
        5 P4 1 ALA SC4 1

        [ bonds ]
        1 2 -- {"custom": 0}
        #meta {"first meta": "a", "first meta 2": "b"}
        2 3 -- {"custom": 1}
        3 4
        #meta {"first meta": 10, "second meta": 20}
        4 5
        """
        lines = textwrap.dedent(lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        interactions = ff.blocks['GLY'].interactions['bonds']
        meta = [interaction.meta for interaction in interactions]
        expected = [
            {'custom': 0},
            {'first meta': 'a', 'first meta 2': 'b', 'custom': 1},
            {'first meta': 'a', 'first meta 2': 'b'},
            {'first meta': 10, 'first meta 2': 'b', 'second meta': 20},
        ]
        assert meta == expected

    @staticmethod
    @pytest.mark.parametrize('meta_line', (
        # Extra column
        '#meta {"a": 0, "b": 1} "plop"',
        # Missing column
        '#meta',
        # The value is not a dict
        '#meta "plop"'
    ))
    def test_invalid_meta(meta_line):
        """
        #meta lines fail when misformatted.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1

        [ bonds ]
        {}
        BB SC1
        """
        lines = textwrap.dedent(lines).format(meta_line)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
            vermouth.ffinput.read_ff(lines, ff)

    @staticmethod
    @pytest.mark.parametrize('section_name', (
        'non-edges', 'patterns', 'features', '!bonds', '!angles'
    ))
    def test_invalid_section_fails(section_name):
        """
        Using the sections that are invalid in a block fails.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1

        [ {} ]
        BB SC1
        """
        lines = textwrap.dedent(lines).format(section_name)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
            vermouth.ffinput.read_ff(lines, ff)

    @staticmethod
    def test_atomname_unique():
        """
        Atom names are unique in a block.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA BB 1
        """
        lines = textwrap.dedent(lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
            vermouth.ffinput.read_ff(lines, ff)

    @staticmethod
    @pytest.mark.parametrize('interaction_line, expected', (
        (
            "BB SC1 1 dist(BB,SC1)",
            ['1', vermouth.molecule.ParamDistance(['BB', 'SC1'])],
        ),
        (
            "BB SC1 2 dist(BB,SC1|.3f) 6",
            ['2', vermouth.molecule.ParamDistance(['BB', 'SC1'], format_spec='.3f'), '6'],
        ),
    ))
    def test_param_effector(interaction_line, expected):
        """
        Parameter effectors are detected and instanciated.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1

        [ bonds ]
        {}
        """
        lines = textwrap.dedent(lines.format(interaction_line))
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        parameters = ff.blocks['GLY'].interactions['bonds'][0].parameters

        assert parameters == expected

    @staticmethod
    def test_param_effector_fail():
        """
        A non existing parameter effector causes the parser to fail.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1

        [ bonds ]
        BB SC1 7 inexistant(BB,SC1)
        """
        lines = textwrap.dedent(lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
            vermouth.ffinput.read_ff(lines, ff)

    @staticmethod
    @pytest.mark.parametrize('interaction_lines, expected', (
        [
            """
            [ dihedrals ]
            BB SC1 SC2 SC3 2 1 2 3 4
            BB SC1 SC2 SC4 1
            """,
            {'impropers': [vermouth.molecule.Interaction(atoms='BB SC1 SC2 SC3'.split(),
                                                         parameters='2 1 2 3 4'.split(),
                                                         meta={})],
             'dihedrals': [vermouth.molecule.Interaction(atoms='BB SC1 SC2 SC4'.split(),
                                                         parameters=['1'],
                                                         meta={})]}
        ],
        [
            """
            [ dihedrals ]
            BB SC1 SC2 SC3 1 1 2 3 4
            BB SC1 SC2 SC4 1
            """,
            {'dihedrals': [vermouth.molecule.Interaction(atoms='BB SC1 SC2 SC3'.split(),
                                                         parameters='1 1 2 3 4'.split(),
                                                         meta={}),
                           vermouth.molecule.Interaction(atoms='BB SC1 SC2 SC4'.split(),
                                                         parameters=['1'],
                                                         meta={})],
             'impropers': []}
        ],
        [
            """
            [ dihedrals ]
            BB SC1 SC2 SC3 2 1 2 3 4
            BB SC1 SC2 SC4 2
            """,
            {'impropers': [vermouth.molecule.Interaction(atoms='BB SC1 SC2 SC3'.split(),
                                                         parameters='2 1 2 3 4'.split(),
                                                         meta={}),
                           vermouth.molecule.Interaction(atoms='BB SC1 SC2 SC4'.split(),
                                                         parameters=['2'],
                                                         meta={})],
             'dihedrals': []}
        ],
    ))
    def test_multiple_impropers(interaction_lines, expected):
        lines = """
        [ moleculetype ]
        XOXO 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P3 1 ALA SC1 2
        3 P2 1 ALA SC2 3
        4 P2 1 ALA SC3 3
        5 P2 1 ALA SC4 3
        """
        lines = textwrap.dedent(lines) + textwrap.dedent(interaction_lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        assert ff.blocks['XOXO'].interactions == expected


class TestLink:
    @staticmethod
    def test_link():
        """
        The minimum link works.
        """
        lines = """
        [ link ]

        [link]
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        assert len(ff.links) == 2

    @staticmethod
    @pytest.mark.parametrize('lines, atoms, interactions, edges', (
        (  # Base case
            """
            [link]
            [bonds]
            BB SC1 1 0.3 100
            """,
            (
                ('BB', {'atomname': 'BB', 'order': 0}),
                ('SC1', {'atomname': 'SC1', 'order': 0}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['BB', 'SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('BB', 'SC1'),),
        ),
        (  # Base case with prefix
            """
            [link]
            [bonds]
            BB +SC1 1 0.3 100
            """,
            (
                ('BB', {'atomname': 'BB', 'order': 0}),
                ('+SC1', {'atomname': 'SC1', 'order': 1}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['BB', '+SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('BB', '+SC1'),),
        ),
        (  # Base case with prefix
            """
            [link]
            [bonds]
            ---BB SC1 1 0.3 100
            """,
            (
                ('---BB', {'atomname': 'BB', 'order': -3}),
                ('SC1', {'atomname': 'SC1', 'order': 0}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['---BB', 'SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('---BB', 'SC1'),),
        ),
        (  # Base case with prefix
            """
            [link]
            [bonds]
            <<BB >SC1 1 0.3 100
            """,
            (
                ('<<BB', {'atomname': 'BB', 'order': '<<'}),
                ('>SC1', {'atomname': 'SC1', 'order': '>'}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['<<BB', '>SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('<<BB', '>SC1'),),
        ),
        (  # Base case with implicit order in the attributes
            """
            [link]
            [bonds]
            BB SC1 {"order": 1} 1 0.3 100
            """,
            (
                ('BB', {'atomname': 'BB', 'order': 0}),
                ('+SC1', {'atomname': 'SC1', 'order': 1}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['BB', '+SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('BB', '+SC1'),),
        ),
        (  # Base case with implicit order in the attributes
            """
            [link]
            [bonds]
            BB SC1 {"order": -2} 1 0.3 100
            """,
            (
                ('BB', {'atomname': 'BB', 'order': 0}),
                ('--SC1', {'atomname': 'SC1', 'order': -2}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['BB', '--SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('BB', '--SC1'),),
        ),
        (  # Base case with implicit order in the attributes
            """
            [link]
            [bonds]
            SC1 SC1 {"order": ">>"} 1 0.3 100
            """,
            (
                ('SC1', {'atomname': 'SC1', 'order': 0}),
                ('>>SC1', {'atomname': 'SC1', 'order': '>>'}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['SC1', '>>SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('SC1', '>>SC1'),),
        ),
        (  # Inhibit edges
            """
            [link]
            [bonds]
            BB SC1 1 0.3 100 {"edge": false}
            """,
            (
                ('BB', {'atomname': 'BB', 'order': 0}),
                ('SC1', {'atomname': 'SC1', 'order': 0}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['BB', 'SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={'edge': False},
                ),
            ]},
            (),
        ),
        (  # Atom already in the link
            """
            [link]
            [atoms]
            BB {"element": "Q"}
            [bonds]
            BB SC1 1 0.3 100
            """,
            (
                ('BB', {'atomname': 'BB', 'order': 0, 'element': 'Q'}),
                ('SC1', {'atomname': 'SC1', 'order': 0}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['BB', 'SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('BB', 'SC1'),),
        ),
        (  # Interaction atoms have attributes
            """
            [link]
            [bonds]
            BB {"custom": 1, "plop": "toto"} SC1 {"toto": "tata"} 1 0.3 100
            """,
            (
                ('BB', {'atomname': 'BB', 'order': 0, 'custom': 1, 'plop': 'toto'}),
                ('SC1', {'atomname': 'SC1', 'order': 0, 'toto': 'tata'}),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['BB', 'SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('BB', 'SC1'),),
        ),
        (  # Atom attributes use a Choice
            """
            [link]
            [bonds]
            BB SC1 {"resname": "GLY|ALA|VAL"} 1 0.3 100
            """,
            (
                ('BB', {'atomname': 'BB', 'order': 0}),
                ('SC1', {
                    'atomname': 'SC1', 'order': 0,
                    'resname': vermouth.molecule.Choice(['GLY', 'ALA', 'VAL']),
                }),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['BB', 'SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('BB', 'SC1'),),
        ),
        (  # Atom attributes use a LinkPredicate defined as link attribute
            """
            [link]
            resname not("VAL")
            [bonds]
            BB SC1 1 0.3 100
            """,
            (
                ('BB', {
                    'atomname': 'BB', 'order': 0,
                    'resname': vermouth.molecule.NotDefinedOrNot('VAL'),
                }),
                ('SC1', {
                    'atomname': 'SC1', 'order': 0,
                    'resname': vermouth.molecule.NotDefinedOrNot('VAL'),
                }),
            ),
            {'bonds': [
                vermouth.molecule.Interaction(
                    atoms=['BB', 'SC1'],
                    parameters=['1', '0.3', '100'],
                    meta={},
                ),
            ]},
            (('BB', 'SC1'),),
        ),
    ))
    def test_link_interactions(lines, atoms, interactions, edges):
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        link = ff.links[0]
        assert tuple(link.nodes(data=True)) == atoms
        assert link.interactions == interactions
        assert (set(frozenset(edge) for edge in link.edges)
                == set(frozenset(edge) for edge in edges))


    @staticmethod
    def test_number_of_links():
        """
        Links are stored in a list so unlike blocks
        they are not overwritten when they are returned
        from the parsers. Thus we should check not only
        that the expected links are in but also how many
        there are.
        """
        lines = """
        [ moleculetype ]
        GLY 1
        [ atoms ]
        1 P4 1 ALA BB 1
        [ link ]
        [ bonds ]
        BB +BB params
        BB SC1 params
        [ angles ]
        BB +BB ++BB params
        [ link ]
        [ bonds ]
        BB SC2 params
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)

        assert len(ff.links) == 2

    @staticmethod
    def test_negative_interactions():
        """
        Prefixing an interaction section name with a ! adds the interactions to
        the ones to delete.
        """
        lines = """
        [ link ]
        [ bonds ]
        BB SC2
        [ !bonds ]
        BB SC1 1 20 3 {"toto": "tata"}
        SC1 {"attr": 8} SC2
        ;[ !custom ]
        ;BB SC1 {"attr_plop": 9} SC2 -- {"plop": 0, "foo": "bar"}
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        link = ff.links[0]
        assert link.interactions == {
            "bonds": [
                vermouth.molecule.Interaction(
                    atoms=['BB', 'SC2'],
                    parameters=[],
                    meta={},
                ),
            ],
        }
        assert link.removed_interactions == {
            "bonds": [
                vermouth.molecule.DeleteInteraction(
                    atoms=['BB', 'SC1'],
                    atom_attrs=[{}, {}],
                    parameters=['1', '20', '3'],
                    meta={'toto': 'tata'},
                ),
                vermouth.molecule.DeleteInteraction(
                    atoms=['SC1', 'SC2'], parameters=[], meta={},
                    atom_attrs=[{'attr': 8}, {}],
                ),
            ],
            #"custom": [
            #    vermouth.molecule.DeleteInteraction(
            #        atoms=['BB', 'SC1', 'SC2'],
            #        atom_attrs=[{}, {'attr_plop': 9}, {}],
            #        parameters=[],
            #        meta={'plop': 0, 'foo': 'bar'},
            #    ),
            #],
        }

    @staticmethod
    def test_features():
        """
        Links accept features.
        """
        lines = """
        [ link ]
        [ features ]
        toto tata
        plop
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        link = ff.links[0]
        assert link.features == {'toto', 'tata', 'plop'}

    @staticmethod
    def test_patterns():
        """
        Links accept patterns.
        """
        lines = """
        [ link ]
        [ patterns ]
        BB {"custom": 0} SC1 SC2 {"plop": "toto"}
        BB SC1 {"custom": "A|BB|CCC"} SC2
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        link = ff.links[0]
        assert link.patterns == [
            [['BB', {'custom': 0}], ['SC1', {}], ['SC2', {'plop': 'toto'}],],
            [
                ['BB', {}],
                ['SC1', {'custom': vermouth.molecule.Choice(['A', 'BB', 'CCC'])}],
                ['SC2', {}],
            ],
        ]

    @staticmethod
    def test_attributes():
        """
        Link attributes are added to the atoms.
        """
        lines = """
        [ link ]
        attr0 "plop"
        attr1 {"foo": "bar"}
        attr2 10
        attr3 "A|BB|CCC"
        attr4 not("stuff")
        [ atoms ]
        XXX {"attr0": "other"}  ; overwrite attr0
        [ bonds ]
        YYY ZZZ
        QQQ {"attr1": 33} WWW  ; overwrite attr1
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        link = ff.links[0]
        # Check the overwritten attributes first
        assert link.nodes['XXX']['attr0'] == 'other'
        assert link.nodes['QQQ']['attr1'] == 33
        assert all(link.nodes[key]['attr0'] == 'plop'
                   for key in ('YYY', 'ZZZ', 'QQQ', 'WWW'))
        assert all(link.nodes[key]['attr1'] == {'foo': 'bar'}
                   for key in ('YYY', 'ZZZ', 'XXX', 'WWW'))
        # Check the others
        assert all(link.nodes[key]['attr2'] == 10 for key in link.nodes)
        assert all(
            link.nodes[key]['attr3'] == vermouth.molecule.Choice(['A', 'BB', 'CCC'])
            for key in link.nodes
        )
        assert all(
            link.nodes[key]['attr4'] == vermouth.molecule.NotDefinedOrNot('stuff')
            for key in link.nodes
        )

    @staticmethod
    def test_moleta():
        """
        The [ molmeta ] section is read.
        """
        lines = """
        [ link ]
        [ molmeta ]
        attr0 10
        attr1 "plop"
        attr2 "A|BB|CCC"
        attr3 not("toto")
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        link = ff.links[0]
        assert link.molecule_meta == {
            'attr0': 10,
            'attr1': 'plop',
            'attr2': vermouth.molecule.Choice(['A', 'BB', 'CCC']),
            'attr3': vermouth.molecule.NotDefinedOrNot('toto'),
        }

    @staticmethod
    def test_non_edges():
        lines = """
        [ link ]
        [ non-edges ]
        XXX YYY {"atomname": "notYYY"}
        ABC +DEF
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        link = ff.links[0]
        assert link.non_edges == [
            ['XXX', {'atomname': 'notYYY', 'order': 0}],
            ['ABC', {'atomname': 'DEF', 'order': 1}],
        ]



class TestModification:
    @staticmethod
    def test_modification():
        """
        The modification section creates a modification.
        """
        lines = """
        [ modification ]
        [ modification ]
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        # 2 modifications were made, but they both had the same name. This
        # should probably raise an error somewhere.
        assert list(ff.modifications.keys()) == ['']

    @staticmethod
    def test_example():
        lines = """
        [ modification ]
        C-ter
        [ atoms ]
        CA{"element": "C"}  ; the space between the name and the attributes is optionnal
        C {"element": "C"}
        O {"element": "O"}
        H {"element": "H", "PTM_atom": true}
        OXT {"element": "O", "PTM_atom": true, "replace": {"atomname": null}}
        [ bonds ]
        O  H  1  0.12  1000
        [ edges ]
        CA C
        C O
        C OXT
        O H
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        vermouth.ffinput.read_ff(lines, ff)
        modification = ff.modifications['C-ter']

        assert modification.name == 'C-ter'
        assert tuple(modification.nodes(data=True)) == (
            ('CA', {'element': 'C', 'PTM_atom': False, 'atomname': 'CA', 'order': 0}),
            ('C', {'element': 'C', 'PTM_atom': False, 'atomname': 'C', 'order': 0}),
            ('O', {'element': 'O', 'PTM_atom': False, 'atomname': 'O', 'order': 0}),
            ('H', {'element': 'H', 'PTM_atom': True, 'atomname': 'H', 'order': 0}),
            ('OXT', {
                'element': 'O',
                'PTM_atom': True,
                'atomname': 'OXT',
                'order': 0,
                'replace': {'atomname': None}
            }),
        )
        assert modification.interactions['bonds'] == [vermouth.molecule.Interaction(atoms=['O', 'H'],
                                                                                    parameters=['1', '0.12', '1000'],
                                                                                    meta={},)]

        assert set(frozenset(edge) for edge in modification.edges) == {
            frozenset(('CA', 'C')),
            frozenset(('C', 'O')),
            frozenset(('H', 'O')),
            frozenset(('C', 'OXT')),
        }


def test_variables():
    """
    Test that the `[ variables ]` section is read as expected.
    """
    lines = """
        [ variables ]
        integer 1
        float 0.3
        string "mass"
        dictionary {"a": 1, "b": {"plop": "toto"}}
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    vermouth.ffinput.read_ff(lines, ff)

    # We test the float first as we want to test its equality with a tolerance.
    # Then we can test that the rest is equal.
    assert np.allclose(ff.variables['float'], 0.3)
    del ff.variables['float']
    assert ff.variables == {'integer': 1, 'string': 'mass',
                            'dictionary': {'a': 1, 'b': {'plop': 'toto'}}}
test_variables()

def test_variables_existing():
    """
    Test that the `[ variables ]` section updates existing variables.
    """
    lines = """
        [ variables ]
        added 1
        updated 2
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    ff.variables['existing'] = 0
    ff.variables['updated'] = 5
    vermouth.ffinput.read_ff(lines, ff)
    assert ff.variables == {'existing': 0, 'added': 1, 'updated': 2}


def test_variables_macro():
    """
    Test that macros are used when defining variables.
    """
    lines = """
        [ macros ]
        int_macro 1
        quote_macro "quoted"
        unquoted_macro unquoted
        int_macro2 2

        [ variables ]
        int_variable $int_macro
        quote_variable $quote_macro
        unquoted_variable $unquoted_macro
        multiple $int_macro$int_macro2
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    vermouth.ffinput.read_ff(lines, ff)
    assert ff.variables == {'int_variable': 1, 'quote_variable': 'quoted',
                            'unquoted_variable': 'unquoted', 'multiple': 12}

def test_variable_first():
    """
    The [ variables ] section must come before blocks, links, or modifications.
    """
    lines = """
        [ macros ]
        int_macro 1
        quote_macro "quoted"
        unquoted_macro unquoted

        [ moleculetype ]
        GLY 2

        [ variables ]
        int_variable $int_macro
        quote_variable $quote_macro
        unquoted_variable $unquoted_macro
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    with pytest.raises(IOError):
        vermouth.ffinput.read_ff(lines, ff)


@pytest.mark.parametrize('lines', (
    # Misformated section title
    """
    [ macros
    """,
    """
    macros]
    """,
    """
    [; macros]
    """,
    # Misformatted attributes
    """
    [ variables ]
    toto {"unfinished": 0
    """,
    """
    [ variables ]
    toto {"unfinished": "plop": 5}}
    """,
    # Misformatted macros
    """
    [ macros ]
    toto tata plop ;  too many columns
    """,
    """
    [ macros ]
    toto  ; not enough columns
    """,
    # Misformatted variables
    """
    [ variables ]
    toto tata plop ;  too many columns
    """,
    """
    [ variables ]
    toto  ; not enough columns
    """,
    # Multiple -- separators
    """
    [ moleculetype ]
    GLY 1

    [ atoms ]
    1 P4 1 ALA BB 1
    2 P4 1 ALA SC1 1

    [ bonds ]
    BB SC1 -- 7 8 -- 9 10
    """,
    # Atom attributes without reference
    """
    [ link ]
    [ patterns ]
    {"atomname": "CA"}
    """,
    # Prefix is inconsistent
    """
    [ link ]
    [ bonds ]
    +-+XXX YYY
    """,
    """
    [ link ]
    [ bonds ]
    ><XXX YYY
    """,
    # Prefix is missing a reference
    """
    [ link ]
    [ bonds ]
    +++ YYY
    """,
    # Invalid order
    """
    [ link ]
    [ atoms ]
    XXX {"order": "invalid"}
    """,
    # Order is inconsistent with the prefix
    """
    [ link ]
    [ atoms ]
    ++XXX {"order": -3}
    """,
    # Mismatch between attributes in different sections
    """
    [ link ]
    [ atoms ]
    XXX {"custom": 0}
    [ bonds ]
    XXX {"custom": 1} YYY
    """,
    """
    [ link ]
    [ bonds ]
    XXX {"custom": 1} YYY
    XXX {"custom": 2} ZZZ
    """,
    # Wrong number of tokens in the link [ atoms ] section
    """
    [ link ]
    [ atoms ]
    XXX
    """,
    """
    [ link ]
    [ atoms ]
    XXX {"custom": 0} "extra"
    """,
    # Wrong number of tokens in the [ molmeta ] section
    """
    [ link ]
    [ molmeta ]
    XXX
    """,
    """
    [ link ]
    [ molmeta ]
    1 2 3
    """,
    # [ molmeta ] section outside a link
    """
    [ molmeta ]
    XXX 2
    """,
    """
    [ moleculetype ]
    ABC   3
    [ molmeta ]
    XXX 2
    """,
    # [ atoms ] section not in an authorized section
    """
    [ atoms ]
    XXX {"custom": 0}
    """,
    """
    [ variables ]
    XXX 10
    [ atoms ]
    ABD {"custom": 0}
    """,
    # Specify edge between unknown atoms
    """
    [ modification ]
    X1
    [ atoms ]
    A {}
    [ edges ]
    A B
    """,
    # Specify edge between unknown atoms
    """
    [ modification ]
    X2
    [ atoms ]
    B {}
    [ edges ]
    A B
    """,
))
def test_misformed_lines(lines):
    """
    Assure that misformed lines cause the parser to stop.
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    with pytest.raises(IOError):
        vermouth.ffinput.read_ff(lines, ff)
