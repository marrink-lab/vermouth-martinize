#!/usr/bin/env python3

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

    !X          : connect active node to node X, which _must_ be present already
                  otherwise, using a name that is already there is an error

    <NAME>      : include brick with given name

    <':NAME>    : include brick with given name and add ' as suffix to nodes

    <NAME@X>    : include brick with given name and add edge between active
                  node and node 'X' of brick
"""


import sys
import string
import networkx as nx


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
        raise GrappaSyntaxError("Matching '}' not found")

    return string[:pos]


def parse_attribute_token(attr):
    """Read attribute token and return corresponding node dictionary"""
    out = {}
    tok = tokenize(attr[1:-1], special=':,', groups=('[]','{}','()'), tgroup=None)
    for token in tok:
        assert next(tok) == ':'
        out[token] = next(tok)
    return out


def expand_nodestring(nodestr):
    """Parse a string like X[1-3,6,8] to list [X1,X2,X3,X6,X8]"""

    if not '[' in nodestr:
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
    G = graphs.get(tag)
    if G is None:
        raise KeyError("Include graph {} not found in graphs.".format(tag))
    mapping = {k: k+lbl for k in G.nodes}
    G = nx.relabel_nodes(G, mapping)
    return G, atpos + lbl


def preprocess(graphstring):
    """
    Expand graphstring 'macros'.

    EXAMPLES:

        /#=1-3/(C#(H#[1-2]))
          --> C1(H11,H12) C2(H21,H22) C3(H31,H32)

        H1 C1(/#=2-4/(C#(H#[1-3]),))
          --> H1 C1(C2(H21,H22,H23),C3(H31,H32,H33),C4(H41,H42,H43))
    """
    tokkie = tokenize(graphstring, special="/", groups=(), tgroup=None)
    out = []
    for token in tokkie:
        if token == '/':
            rng = next(tokkie)

            if next(tokkie) != '/':
                err = 'Matching "/" not found during preprocessing'
                raise GrappaSyntaxError(err)

            subst, values = rng.split('=')
            values = [val.split('-') for val in values.split(',')]

            form = next(tokkie)

            if next(tokkie) != '/':
                err = 'Matching "/" not found during preprocessing'
                raise GrappaSyntaxError(err)

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
             special='@(),-=!', groups=('{}', '<>'), tgroup='[]'):
    """
    Parse a token string and tokenize it, return tokenlist
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

    tokens = tokenize(graphstring)
    directives = '(),@-=<>{}'
    #print(graphstring)
    #print(tokens)

    G = nx.Graph()
    active = None
    parent = []
    for token in tokens:

        if token == "!":
            # Make connection to already available node (next token)
            token = next(tokens)
            if token not in G:
                print(G.nodes)
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
#                print("Adding stub to", node, ":", G.nodes[node]['stub'])
            else:
                # Token is node or nodes
                nodes = expand_nodestring(token)
                if node is None:
#                    print('Unrooted nodes:', *nodes)
                    G.add_nodes_from(nodes)
                else:
                    G.add_edges_from((node, n) for n in nodes)
#                        print("Edge:", node, "to", n)
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
#            print("End of branching: active:", active[-1])

        elif token == ',':
            # Switch to next branch
            active = None

        elif token == '@':
            # Set node as active
            active = next(tokens)
#            print("Setting active:", active)

        elif token == '-':
            # Remove node
            G.remove_node(next(tokens))

        elif token == '=':
            # Rename active node
            G = nx.relabel_nodes(G, {active: next(tokens)})

        elif token[0] == '<':
            # Include graph from graphs and relabel nodes according to tag
            # <tag:graphname@node>
            B, at = include_graph(graphs, token[1:-1])
#            print("Including graph from", token, ":", *B.nodes)
            G.add_nodes_from(B.nodes)
            G.add_edges_from(B.edges)
            if active is not None:
                G.add_edge(active, at)
            elif parent and parent[-1] is not None:
                G.add_edge(parent[-1], at)

        elif token[0] == '{':
            # Set attributes to active node
            print("Setting attributes at active atom:", parse_attribute_token(token))
            G.nodes[active].update(parse_attribute_token(token))

    #print(G.nodes)
    #print(G.edges)
    return G


# TODO: Move amino acid graph strings to data folder
# TODO: Replace this function with a proper test
def amino_acid_test():
    """Test amino acid grappa set"""

    stuff = """
[ BB ] N(H,.) CA(HA,.) C(O1,.) @CA {chiral:(N,C,HA)}
[ GLY / glycine       / AA::G ] <BB> @HA =HA1 @CA HA2
[ ALA / alanine       / AA::A ] <BB> @CA CB(HB[1-3])
[ ASP / aspartate     / AA::D ] <BB> @CA CB(HB1,HB2) CG(OD1,OD2) 
[ ASN / asparagine    / AA::N ] <BB> @CA CB(HB1,HB2) CG(OD1) ND2(HD21,HD22)
[ SER / serine        / AA::S ] <BB> @CA CB(HB1,HB2) OG HG
[ CYS / cysteine      / AA::C ] <BB> @CA CB(HB1,HB2) SG HG
[ MET / methionine    / AA::M ] <BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(HD1,HD2) SD CE(HE[1-3])
[ THR / threonine     / AA::T ] <BB> @CA CB (OG1 HG1) CG2 (HG2[1-3])
[ VAL / valine        / AA::V ] <BB> @CA CB(HB,CG1(HG1[1-3]),CG2(HG2[1-3])) 
[ ILE / isoleucince   / AA::I ] <BB> @CA CB(HB,CG2(HG2[1-3])) CG1(HG1[1-2]) CD(HD[1-3])
[ LEU / leucine       / AA::L ] <BB> @CA CB(HB1,HB2) CG(HG1,CD1(HD1[1-3]),CD2(HD2[1-3]))
[ GLU / glutamate     / AA::E ] <BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(OE1,OE2) 
[ GLN / glutamine     / AA::Q ] <BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(OE1) NE2(HE21,HE22)
[ PRO / proline       / AA::P ] <BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(HD1,HD2) !C 
[ HIS / histidine     / AA::H ] <BB> @CA CB(HB1,HB2) CG CE1(HE1) ND1 CE1(HE1) NE2(HE2 {pKa:6.04}) CD2(HD2) !CG
[ PHE / phenylalanine / AA::F ] <BB> @CA CB(HB1,HB2) CG CD1(HD1) CE1(HE1) CZ(HZ) CE2(HE2) CD2(HD2)
[ TYR / tyrosine      / AA::Y ] <PHE> -HZ @CZ OH HH {pKa:10.10}
[ LYS / lysine        / AA::K ] <BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(HD1,HD2) CE(HE1,HE2) NZ(HZ1,HZ2,HZ3)
[ ARG / arginine      / AA::R ] <BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(HD1,HD2) NE(HE) CZ(NH1(HH11,HH12),NH2(HH21,HH22 {pKa:12.10}))
[ TRP / tryptophane   / AA::W ] <BB> @CA CB(HB1,HB2) 
"""

    graphs = {}
    for line in stuff.split('\n'):
        if line.strip():
            label, graphstring = line.split(']', 1)
            name = label[1:].split()[0]
            print("\n", name, '-->', graphstring)
            graphs[name] = process(graphstring, graphs)
            print(graphs[name].nodes)
    return graphs


def main(args):

    graphs = amino_acid_test()

    if len(args) > 1:
        P = process( preprocess(" ".join(args[1:])), graphs)
        print(P.nodes)
        print(P.edges)

    return 0


if __name__ == "__main__":
    exitcode = main(sys.argv)
    sys.exit(exitcode)
