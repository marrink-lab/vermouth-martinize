graph [
    name "BB"
    node [
        id 0
        label "N"
         ]
    node [
        id 1
        label "C"
         ]
    node [
        id 2
        label "O"
         ]
    node [
        id 3
        label "CA"
         ]
    edge [ source 0 target 3 ]
    edge [ source 1 target 2 ]
    edge [ source 1 target 3 ]
]
graph [
    name "SC1"
    node [
        id 0
        label "CB"
         ]
    node [
        id 1
        label "CG"
         ]
    node [
        id 2
        label "CD1"
         ]
    node [
        id 3
        label "CD2"
         ]
    edge [ source 0 target 1 ]
    edge [ source 1 target 2 ]
    edge [ source 1 target 3 ]
]
