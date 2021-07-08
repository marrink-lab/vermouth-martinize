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

Maximum common induced subgraph
+++++++++++++++++++++++++++++++
The maximum common induced subgraph between :math:`G` and :math:`H` is the
largest graph :math:`J` such that :math:`J \precsim G` and :math:`J \precsim H`.
Commonly the answer is given as a general mapping between :math:`G` and
:math:`H`.

Isomorphism
-----------
