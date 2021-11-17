Graph algorithms
================
Vermouth describes molecules and molecular fragments as graphs where atoms are
nodes and connections between them (e.g. bonds) are edges. This allows us to use
the *connectivity* to identify which atom is which, meaning we are no longer
dependent on atom names.

Definitions
-----------
Graph
+++++
A graph :math:`G = (V, E)` is a collection of nodes (:math:`V`) connected by
edges (:math:`E`): :math:`e_{ij} = (v_i, v_j) \in E`. In undirected graphs
:math:`e_{ij} = e_{ji}`. Unless we specify otherwise all graphs used in vermouth
are undirected. The size of a graph is equal to the number of nodes:
:math:`|G| = |V|`.

Subgraph
++++++++
Graph :math:`H = (W, F)` is a subgraph of graph :math:`G = (V, E)` if:

.. math::
   |H| &< |G|\\
   W &\subset V\\
   e_{ij} &\in F \quad \forall e_{ij} \in E\\
   e_{ij} &\notin F \quad \forall e_{ij} \notin E\\

This means that all nodes in :math:`H` are in :math:`G`, and that nodes are
connected in :math:`H` if and only if they are connected in :math:`G`.

Graph isomorphism
+++++++++++++++++
A graph isomorphism :math:`m` between graphs :math:`H = (W, F)` and
:math:`G = (V, E)` is a bijective mapping :math:`m: V \mapsto W` such that the
following conditions hold:

.. math::
   |H| &= |G|\\
   m(v) &\simeq v \quad &\forall v \in V\\
   (m(v_i), m(v_j)) &\simeq (v_i, v_j) \quad &: (m(v_i), m(v_j)) \in F \enspace \forall (v_i, v_j) \in E

This means that every node in :math:`G` maps to exactly one node in :math:`H`
such that all connected nodes in :math:`G` are connected in :math:`H`. Note that
labels/attributes on nodes and edges (such as element or atom name) can affect
the equivalence criteria.

Subgraph isomorphism
++++++++++++++++++++
A subgraph isomorphism is a :ref:`graph_algorithms:graph isomorphism`, but
without the constraint that :math:`|H| = |G|`. Instead, :math:`|H| <= |G|` if
:math:`H` is subgraph isomorphic to :math:`G`.

Induced subgraph isomorphism
++++++++++++++++++++++++++++
As :ref:`graph_algorithms:subgraph isomorphism` with the added constraint that
equivalent nodes not connected in :math:`G` are not connected in :math:`H`:

.. math::
   (m(v_i), m(v_j)) \notin F \quad \forall (v_i, v_j) \notin E

We denote :math:`H` being induced subgraph isomorphic to :math:`G` as
:math:`H \precsim G`.

It is important to note that a path graph is *not* subgraph isomorphic
to the corresponding cycle graph of the same size. For example, n-propane is not
subgraph isomorphic to cyclopropane!

Maximum common induced subgraph
+++++++++++++++++++++++++++++++
The maximum common induced subgraph between :math:`G` and :math:`H` is the
largest graph :math:`J` such that :math:`J \precsim G` and :math:`J \precsim H`.
Commonly the answer is given as a general mapping between :math:`G` and
:math:`H`.

Isomorphism
-----------
Vermouth and martinize2 identify atoms by connectivity, generally combined with
a constraint on element or atom name. We do this using either a
:ref:`graph_algorithms:maximum common induced subgraph` (during the
:ref:`martinize2_workflow:Repair graph` step) or a
:ref:`graph_algorithms:induced subgraph isomorphism` (the other steps). In all
these cases we effectively find how nodes in the molecule we're working on match
with nodes in our reference graphs, such as :ref:`blocks <data:block>`.

During the :ref:`martinize2_workflow:Repair graph` step there are two, related,
complications: 1) we need a "best" overlay, where as many atom names match as
possible; and 2) There can be very many (equivalent) possible
overlays/isomorphisms. Let's address the second concern first. As example we'll
look at the automorphisms (= self-isomorphism, i.e. how does a graph fit on
itself) of propane (``CH3-CH2-CH3``).

There are 2 isomorphisms for the carbons:
:math:`C_\alpha-C_\beta-C_\gamma \mapsto C_\alpha-C_\beta-C_\gamma` and
:math:`C_\alpha-C_\beta-C_\gamma \mapsto C_\gamma-C_\beta-C_\alpha`. Similarly,
there are 2 isomorphisms for the central methylene group:
:math:`H_1-C_\beta-H_2 \mapsto H_1-C_\beta-H_2` and
:math:`H_1-C_\beta-H_2 \mapsto H_2-C_\beta-H_1`. Each terminal methyl group
however, has 6 unique isomorphisms!

.. math::
   H_1H_2H_3 \mapsto (H_1H_2H_3, H_1H_3H_2, H_2H_1H_3, H_3H_1H_2, H_2H_3H_1, H_3H_2H_1)

This means that in total, propane, a molecule consisting of 11 atoms, has
:math:`2 (carbons) \times 2 (methylene) \times 6 (methyl) \times 6 (methyl) = 144`
automorphisms! Now imagine how this scales for a lipid. Clearly this spirals out
of control very quickly, and it is generally unfeasible to generate all possible
isomorphisms [#]_.

Luckily for us however, we're not interested in finding all these isomorphisms,
since we can consider most of these to be equivalent. For our use case it
doesn't matter whether :math:`H_1` maps to :math:`H_1` or :math:`H_2` so long as
:math:`H_1` and :math:`H_2` are equivalent. There is one catch however: we need
to find the isomorphism where most atom names match. We can achieve this by
preferentially using nodes with a lower index [#]_ when given a choice between
symmetry equivalent nodes. The [ISMAGS]_ algorithm does exactly this: it
calculates symmetry unique isomorphisms preferentially using nodes with a
smaller index.

Note that this problem only comes up when your graphs are (very) symmetric. In
all other steps we constrain the isomorphism such that nodes are only considered
equal if their atom names match. Since atom names are generally unique, this
means that this problem is sidestepped completely. The only place where we
cannot do this is during the :ref:`martinize2_workflow:Repair graph` step, since
at that point we cannot assume that the atoms names in our molecule are correct.

.. [#] This problem gets *even* worse when trying to find the
   :ref:`graph_algorithms:maximum common induced subgraph`.
.. [#] In other words, we impose an ordering on the nodes in the graph. We do
   this by ordering the nodes based on whether there is a node with a
   corresponding atom name in the reference and subsequently sorting by atom name.
.. [ISMAGS] M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle, M. Pickavet, The Index-Based Subgraph Matching Algorithm with General Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph Enumeration, PLoS One. 9 (2014) e97896. doi:10.1371/journal.pone.0097896.