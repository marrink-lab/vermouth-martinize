if [[ "$#" != 2 ]]
then
    echo "Arguments must be <itp> <aa-pdb>" >&2
    exit 1
fi

itp=$1
pdb=$2

{
vmd -dispdev text -eofexit << EOF
source addDihedral.tcl
addDihedral $itp $pdb
EOF
} | grep -v Info | grep -v vmd | tail -n+4 >> $itp
