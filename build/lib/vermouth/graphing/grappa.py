#!/usr/bin/env python3
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
GRAPh PArser

Build graphs from string definition:

Grappa allows constructing a graph using a linear sequence of tokens,
the grappa string. A token can be a node name, a definition of a series
of nodes or a special character/directive. The principle behind the
grappa string is that the graph is extended from an active node. A new
node is connected to the active node. Using @ the active node can be
switched, and using () branches can be specified on an active node. A
complete description of the grappa minilanguage is given below:

  Grappa string Rules:

    name        : add node with name, with edge to active node
                  (none at start, active parent at start of branch)
    -name       : remove node with name

    @name       : select name as active node

    (           : set active node as active parent (start branching)

    ,           : switch to new branch at active parent

    )           : set active parent to active node

    =nameB      : rename active node (keep edges)

    {attr=val}  : set attribute on active node (can be
                  attributes like:  element, charge, valence, stubs
                  element is set to FIRST LETTER of name,
                  unless specified as attribute
                  attribute chiral has tuple of three nodes,
                  which define chirality according to right-hand rule

    !X          : connect active node to node X, which _must_ be present
                  already. Otherwise, using a name that is already there is an
                  error

    <NAME>      : include brick with given name

    <':NAME>    : include brick with given name and add ' as suffix to nodes

    <NAME@X>    : include brick with given name and add edge between active
                  node and node 'X' of brick

    /#=1-20/C#(H#[1-2])/ : Expand by repetion and substitution according to
                           range specified

    (/#=A-D/C#(H#[1-3]),/) : Expand to multiple branches
"""

import string

import networkx as nx

from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


DIRECTIVES = '@(),-=!'
GROUPS = ('{}', '<>')
TGROUP = '[]'


class GrappaSyntaxError(Exception):
    """Syntax of grappa string was invalid"""


def find_matching(symbols, string):
    """Find matching symbol in a series with possible nesting."""
    nesting = 0
    pos = 0
    while pos < len(string):
        if string[pos] == symbols[0]:
            nesting += 1
        elif string[pos] == symbols[1]:
            nesting -= 1
        pos += 1
        if not nesting:
            break
    else:
        raise GrappaSyntaxError("Matching '{}' not found".format(symbols[1]))
    return string[:pos]


def parse_attribute_token(attr):
    """Read attribute token and return corresponding node dictionary"""
    out = {}
    tok = tokenize(attr[1:-1], special=':,',
                   groups=('[]', '{}', '()'), tgroup=None)
    for token in tok:
        assert next(tok) == ':'
        out[token] = next(tok)
    return out


def expand_nodestring(nodestr):
    """
    Parse a string like X[1-3,6,8] to list [X1,X2,X3,X6,X8].
    X[20-25] is invalid, but X[w-z] is valid.
    """

    if '[' not in nodestr:
        return [nodestr]

    openbra = nodestr.find('[')
    closebra = nodestr.rfind(']')
    if closebra == -1:
        err = 'Matching square bracket not found in node list definition ({})'
        raise GrappaSyntaxError(err.format(nodestr))

    base = nodestr[:openbra]
    nodes = []
    what = [item.split('-') for item in nodestr[openbra+1:closebra].split(',')]
    for thing in what:
        if len(thing) == 1:
            nodes.append(thing[0].strip())
        # And here is why X[20-25] is invalid
        elif len(thing[0]) == 1 and len(thing[1]) == 1:
            for val in range(ord(thing[0]), ord(thing[1]) + 1):
                nodes.append(base + chr(val))
        else:
            err = 'Malformed range in node string ({}-{})'
            raise GrappaSyntaxError(err.format(*thing))

    return nodes


def include_graph(graphs, tag):
    """
    Prepare include graph from graphs, according to tag
    """
    atpos = tag.rfind('@')
    if atpos > 0:
        tag, atpos = tag[:atpos], tag[atpos+1:]
    else:
        atpos = ""
    lbl = tag.find(':')
    if lbl > 0:
        lbl, tag = tag[:lbl], tag[lbl+1:]
    else:
        lbl = ""
    try:
        G = graphs[tag]
    except KeyError:
        raise KeyError('Include graph {} not found in graphs.'.format(tag))
    mapping = {k: k+lbl for k in G.nodes}
    G = nx.relabel_nodes(G, mapping)
    return G, atpos + lbl


def preprocess(graphstring):
    """
    Expand graphstring 'macros'.

    EXAMPLES:

        /#=1-3/(C#(H#[1-2]))/
          --> C1(H11,H12) C2(H21,H22) C3(H31,H32)

        H1 C1(/#=2-4/(C#(H#[1-3]),)/)
          --> H1 C1(C2(H21,H22,H23),C3(H31,H32,H33),C4(H41,H42,H43))
    """
    tokkie = tokenize(graphstring, special="/", groups=(), tgroup=None)
    out = []
    for token in tokkie:
        if token == '/':
            rng = next(tokkie)

            # The range should be simple (single), so the next token must be '/'
            nexttok = next(tokkie)
            if nexttok != '/':
                err = 'Matching "/" not found during preprocessing (found {})'.format(nexttok)
                raise GrappaSyntaxError(err)

            subst, values = rng.split('=')
            values = [val.split('-') for val in values.split(',')]

            form = [next(tokkie)]

            nexttok = next(tokkie)
            try:
                while nexttok != '/':
                    form.append(nexttok)
                    nexttok = next(tokkie)
            except StopIteration:
                err = 'Matching "/" not found during preprocessing (found {})'.format(nexttok)
                raise GrappaSyntaxError(err)
            
            form = " ".join(form)

            for val in values:
                if len(val) == 1:
                    out.append(form.replace(subst, val[0]))
                elif len(val[0]) == 1 and len(val[1]) == 1:
                    for ordi in range(ord(val[0]), ord(val[1]) + 1):
                        out.append(form.replace(subst, chr(ordi)))
                else:
                    for num in range(int(val[0]), int(val[1])+1):
                        out.append(form.replace(subst, str(num)))
        else:
            out.append(token)

    return " ".join(out)


def tokenize(tokenstring, skip=string.whitespace,
             special=DIRECTIVES, groups=GROUPS, tgroup=TGROUP):
    """
    Parse a token string and tokenize it, yielding tokens
    """

    # This is a simplified tokenizer..
    i = -1
    while i + 1 < len(tokenstring):
        i += 1
        here = tokenstring[i]

        if here in skip:
            continue

        if here in special:
            yield here
            continue

        broken = False
        for grp in groups:
            if here == grp[0]:
                here = find_matching(grp, tokenstring[i:])
                yield here
                i += len(here)
                broken = True
                break
        if broken:
            continue

        j = i + 1
        bracket = False
        while j < len(tokenstring):
            char = tokenstring[j]
            if tgroup and char == tgroup[0]:
                bracket = True
            elif (not bracket and 
                  (char in skip or char in special)):
                break
            j += 1
            if tgroup and char == tgroup[1]:
                break

        yield tokenstring[i:j]
        i = j-1


def process(graphstring, graphs={}):
    """
    Parse a graph string construct the corresponding graph.
    """

    tokens = tokenize(preprocess(graphstring))
    directives = '(),@-=<>{}'

    G = nx.Graph()
    active = None
    parent = []
    for token in tokens:

        if token == "!":
            # Make connection to already available node (next token)
            token = next(tokens)
            if token not in G:
                raise IndexError("Token missing in graph: !{}".format(token))

        if token[0] not in directives:
            # Then it's a node (or group of nodes)

            if active is not None:
                # Extending at active node
                node = active
            elif parent:
                # Branching from parent node
                node = parent[-1]
            else:
                # Unrooted: Starting new (unconnected) graph
                node = None

            if token == '.' and node is not None:
                # Adding stub (or stub branch) to active node
                G.nodes[node]['stub'] = G.nodes[node].get('stub', 0) + 1
                LOGGER.debug("Adding stub to {}: {}", node, G.nodes[node]['stub'],
                             type='grappa')
            else:
                # Token is node or nodes
                nodes = expand_nodestring(token)
                if node is None:
                    LOGGER.debug('Unrooted nodes: {}', nodes, type='grappa')
                    G.add_nodes_from(nodes)
                else:
                    G.add_edges_from((node, n) for n in nodes)
                    LOGGER.debug("Edge: {} to {}", node, nodes, type='grappa')
                active = nodes[-1]
            continue

        # Directives:
        if token == '(':
            # Start branching
            parent.append(active)
            active = None

        elif token == ')':
            # End branch(es) - switch to active parent
            active = parent.pop()
            LOGGER.debug("End of branching: active: {}", active[-1], type='grappa')

        elif token == ',':
            # Switch to next branch
            active = None

        elif token == '@':
            # Set node as active
            active = next(tokens)
            LOGGER.debug("Setting active: {}", active, type='grappa')

        elif token == '-':
            # Remove node
            G.remove_node(next(tokens))

        elif token == '=':
            # Rename active node
            G = nx.relabel_nodes(G, {active: next(tokens)})

        elif token[0] == '<':
            # Include graph from graphs and relabel nodes according to tag
            # <tag:graphname@node>. include_graph relabels its nodes.
            B, at = include_graph(graphs, token[1:-1])
            LOGGER.debug("Including graph from {}: {}", token, B.nodes, type='grappa')
            G.add_nodes_from(B.nodes.items())
            G.add_edges_from(B.edges())
            if active is not None:
                G.add_edge(active, at)
            elif parent and parent[-1] is not None:
                G.add_edge(parent[-1], at)

        elif token[0] == '{':
            # Set attributes to active node
            LOGGER.debug("Setting attributes at active atom: {}",
                         parse_attribute_token(token), type='grappa')
            G.nodes[active].update(parse_attribute_token(token))

    return G
