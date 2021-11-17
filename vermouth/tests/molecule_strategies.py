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

import hypothesis.strategies as st
from hypothesis_networkx import strategy as hnst
import vermouth
import vermouth.molecule
from vermouth.molecule import Interaction, Molecule, Block, Link, DeleteInteraction


# pylint: disable=no-value-for-parameter

# This is a strategy that creates "sane" names. It's currently used to generate
# interaction type names, and molecule names. The produced strings will not
# contain characters from the unicode categories C and Z, which contain
# control characters and separators, respectively.
SANE_NAME_STRATEGY = st.text(st.characters(blacklist_categories=('C', 'Z')), min_size=1)


@st.composite
def attribute_dict(draw, min_size=0, max_size=None, max_depth=1):
    """
    Strategy that builds an attribute dictionary for meta or atoms.

    Parmeters
    ---------
    draw:
        Internal for hypothesis.
    min_size: int
        The minimum number of key-value pairs.
    max_size: int, optional
        The maximum number of key-value pairs. If set to `None` (default),
        there is no bound set for the maximum size of the dictionary in the
        same manner as :func:`hypothesis.strategies.dictionaries`.
    max_depth: int
        The strategy is recursive so that the values can be attribute
        dictionaries or lists containing attribute dictionaries. This argument
        sets the maximum depth of the recursion.

    Returns
    -------
    hypothesis.searchstrategy.lazy.LazyStrategy
    """
    keys = st.one_of(st.integers(), st.none(), st.text())
    bases = [st.none(), st.text(), st.integers(), st.floats()]
    if max_depth > 0:
        bases.append(attribute_dict(max_size=1, max_depth=max_depth - 1))
    lists = st.lists(st.one_of(*bases), max_size=2)
    values = st.one_of(*bases, lists)
    return draw(st.dictionaries(keys, values, min_size=min_size, max_size=max_size))


@st.composite
def parameter_effectors(draw):
    """
    Strategy that builds a :class:`~vermouth.molecule.LinkParameterEffector`.

    The strategy chooses one possible parameter effector class, and creates an
    instance with random keys and format spec.
    """
    possible_effectors = [
        vermouth.molecule.ParamDistance,
        vermouth.molecule.ParamAngle,
        vermouth.molecule.ParamDihedral,
        vermouth.molecule.ParamDihedralPhase,
    ]
    effector = draw(st.sampled_from(possible_effectors))
    n_keys = effector.n_keys_asked
    keys = [draw(st.text(min_size=1, max_size=4)) for _ in range(n_keys)]
    possible_formats = [None, '.2f', '3.0f']
    format_spec = draw(st.sampled_from(possible_formats))

    return effector(keys, format_spec=format_spec)


@st.composite
def random_interaction(draw, graph, natoms=None,
                       interaction_class=Interaction, attrs=False):
    """
    Strategy that builds an interaction or interaction-like object.

    By default, the strategy builds an instance of
    :class:`~vermouth.molecule.Interaction`. The strategy can also build a
    :class:`~vermouth.molecule.DeleteInteraction` by giving `DeleteInteraction`
    to the `interaction_class` argument and `True` to the `attrs` one.

    Parameters
    ----------
    draw:
        Internal for hypothesis.
    graph: networkx.Graph
        Graph/molecule from which nodes will be drawn.
    natoms: int, optional
        The number of atoms involved in the interaction. If set to `None`
        (default), a random number between 1 and 4 (included) is used.
    interaction_class: type
        The class to use for the interaction.
    attrs: bool
        Wether ot not to build an `atom_attrs` argument for the interaction.
        This should be `False` if the interaction is a
        :class:`~vermouth.molecule.Interaction`, and `True` if it is a
        :class:`~vermouth.molecule.DeleteInteraction`.

    Returns
    -------
    hypothesis.searchstrategy.lazy.LazyStrategy
    """
    if natoms is None:
        natoms = draw(st.integers(min_value=1, max_value=4))
    if graph:
        atoms = tuple(draw(st.sampled_from(list(graph.nodes))) for _ in range(natoms))
    else:
        atoms = []
    parameters = st.lists(elements=st.one_of(st.text(), parameter_effectors()))
    meta = draw(st.one_of(st.none(), attribute_dict()))
    if attrs:
        atom_attrs = tuple(draw(attribute_dict(max_size=3)) for _ in atoms)
        return interaction_class(
            atoms=atoms,
            atom_attrs=atom_attrs,
            parameters=parameters,
            meta=meta,
        )
    return interaction_class(atoms=atoms, parameters=parameters, meta=meta)


@st.composite
def interaction_collection(draw, graph,
                           interaction_class=Interaction, attrs=False):
    """
    Strategy that builds a collection of interaction-like instances.

    The collection is a dictionary with any string as key, and a list of
    :class:`~vermouth.molecule.Interaction` or
    :class:`~vermouth.molecule.DeleteInteraction`.

    The parameters are passed to :func:`random_interaction`.

    See Also
    --------
    random_interaction
    """
    result = {}
    ninteraction_types = draw(st.integers(min_value=0, max_value=2))
    for _ in range(ninteraction_types):
        ninteractions = draw(st.integers(min_value=0, max_value=2))
        type_name = draw(SANE_NAME_STRATEGY)
        if type_name not in result:
            result[type_name] = []
        for _ in range(ninteractions):
            interaction = draw(random_interaction(
                graph, interaction_class=interaction_class, attrs=attrs,
            ))
            result[type_name].append(interaction)
    return result


@st.composite
def random_molecule(draw, molecule_class=Molecule, max_nodes=5, max_meta=None):
    """
    Strategy that builds a molecule.

    Parameters
    ----------
    draw:
        Internal for hypothesis.
    molecule_class: type
        The class of molecule to build, :class:`vermouth.molecule.Molecule` by
        default.

    Returns
    -------
    hypothesis.searchstrategy.lazy.LazyStrategy
    """
    graph = draw(hnst.graph_builder(
        max_nodes=max_nodes,
        node_data=attribute_dict(max_size=3),
        edge_data=attribute_dict(max_size=2),
    ))
    meta = draw(attribute_dict(max_size=max_meta))
    nrexcl = draw(st.one_of(st.none(), st.integers()))
    molecule = molecule_class(graph, meta=meta, nrexcl=nrexcl)

    molecule.interactions = draw(interaction_collection(molecule))

    return molecule


@st.composite
def random_block(draw, block_class=Block, max_nodes=5, max_meta=None):
    """
    Strategy that builds a block.

    Parameters
    ----------
    draw:
        Internal for hypothesis.
    block_class: type
        The class of block to build, :class:`vermouth.molecule.Block` by
        default.

    Returns
    -------
    hypothesis.searchstrategy.lazy.LazyStrategy
    """
    block = draw(random_molecule(molecule_class=block_class,
                                 max_nodes=max_nodes, max_meta=max_meta))
    block.name = draw(st.one_of(st.none(), st.text()))
    return block


@st.composite
def random_link(draw, max_nodes=5, max_meta=None):
    """
    Strategy that builds a link.

    Parameters
    ----------
    draw:
        Internal for hypothesis.
    link_class: type
        The class of block to build, :class:`vermouth.molecule.Link` by
        default.

    Returns
    -------
    hypothesis.searchstrategy.lazy.LazyStrategy
    """
    link = draw(random_block(block_class=Link,
                             max_nodes=max_nodes, max_meta=max_meta))
    link.removed_interactions = draw(interaction_collection(
        link, interaction_class=DeleteInteraction, attrs=True,
    ))
    link.molecule_meta = draw(attribute_dict(max_size=4))
    link.non_edges = draw(st.lists(
        st.tuples(st.text(min_size=1, max_size=4), attribute_dict(max_size=4)),
        max_size=4,
    ))
    link.patterns = draw(st.lists(
        st.tuples(st.text(min_size=1, max_size=3), attribute_dict(max_size=2)),
        max_size=3,
    ))
    link.features = set(draw(st.lists(st.text(max_size=4), max_size=4)))
    return link
