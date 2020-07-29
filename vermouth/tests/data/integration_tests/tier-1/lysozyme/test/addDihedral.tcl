#########################################################################################
#
# SYNOPSIS
#		addDihedral topology pdbcode
#
# DESCRIPTION
#		this VMD script generates the topology for SC - BB - BB - SC dihedrals. 
#		- loads the pdb file "pdbcode"
#		- reads the Martini topology file "topology"
# 		- it measures the dihedrals in the atomistic structure
#		- prints the topology for extended structuers:
#			- SC - BB - BB - SC dihedrals
#			- SC - BB - BB and BB - BB - SC restricted angle potentials
# EXAMPLE
# 		addDihedral "Protein_A.itp" "1mai"
#
# AUTHOR: Florian Herzog <florian.herzog@hest.ethz.ch
#
# 04/30/2015
#
# CHANGES IN THE STANDARD CODE
#
# - Now the program add SC-BB-BB-SC diedrals and BB-BB-SC angles for all secondary structures
# - AUTHOR: Paulo Cesar Telles de Souza
# LAST CHANGE: 05/15/2017
#
#########################################################################################


proc addDihedral {topology pdbcode} {
	# PARAMETERS
	set b 100						;# force constant for dihedral: V = b(1+cos(phi-a))
	set k1 5						;# force constant for SBB ReB angle
	set k2 5						;# force constant for BBS ReB angle
	set theta1 100					;# theta0 for the SBB ReB angle
	set theta2 100					;# theta0 for the BBS ReB angle
	set n_reb 10					;# function number for the restricted bending potential (might change between gromacs versions)
	
	set ff	"martini"				;# force-field: either martini or elnedyn
	
	# LOAD PDB FILE $pdbcode. Remove these lines if you have your atomistic structure loaded in VMD as "top"
	puts "load pdb file $pdbcode"
	mol new $pdbcode
	
	# READ TOPOLOGY: secondary structure, index and resid in $topology
	set fp [open $topology r]
	set file_data [read $fp]
	close $fp
	set data [split $file_data "\n"]
	set atom_section false
	for {set i 0} {$i < [llength $data]} {incr i} {
		set line [lindex $data $i]
	
		if {[regexp {^\; Secondary Structure} $line]} {
			set ss [lindex $data [expr $i+1]]    
			set ss [regexp {([A-Z,1-9]+)} $ss tmp]
			set ss [split $tmp {}]
		}
		if {[regexp {^\; Sequence} $line]} {
			set seq [lindex $data [expr $i+1]]    
			set seq [regexp {([A-Z,1-9]+)} $seq tmp2]
			set seq [split $tmp2 {}]
		}
	
		if {$atom_section && ([llength $line] == 9 || [llength $line] == 10)} {
			set resid [lindex $line 2]
			if {[lindex $line end] == [lindex $ss [expr $resid-1]]} {
				set index [lindex $line 0]
				set type [lindex $line 4]
				lappend idx $index
				lappend rsd $resid
				lappend typ $type
			} else {
				puts "ERROR reading the topology $topology"
			}
		
		}
	
		if {[regexp {^\[ atoms} $line]} {
			set atom_section true
		}
	
		if {[regexp {^\[ bonds} $line]} {
			set atom_section false
		}
	}
	# MEASURE DIHEDRALS BB BB BB for atomistic structure
	#	for GLY / ALA   -> dihedral = 0
	# NOTE: changes the position of CA and CB atoms in atomistic structures !!

	set ext [atomselect "top" "name CA"]

	set lext [$ext get resid]
	set n [$ext num]

	# place CA at center of mass of BB definition
	# 	and CB at center of mass of SC1 definition
	set numframes [molinfo "top" get numframes]
 
	puts "moving CA and CB to center of mass of BB, SC1 bead"
	foreach res $lext {
		set resname [[atomselect "top" "resid $res and name CA"] get resname]
		# puts "$resname $res"
		if {$ff eq "martini"} {
			set bb "N CA C O H H1 H2 H3 O1 O2"
		} elseif {$ff eq "elnedyn"} {
			set bb "CA"
		} else {
			puts "ERROR: no proper forcefield definition: $ff"
		}
		set sc ""
		if {$resname eq "ALA"} { 
			if {$ff eq "martini" } {
				set bb [concat $bb "CB"]
			}
			set sc ""
		} elseif {$resname eq "CYS"} {
			set sc "CB SG"
		} elseif {$resname eq "ASP"} {
			set sc "CB CG OD1 OD2"
		} elseif {$resname eq "GLU"} {
			set sc "CB CG CD OE1 OE2"
		} elseif {$resname eq "PHE"} {
			set sc "CB CG CD1 HD1"
		} elseif {$resname eq "GLY"} {
			set sc ""
		} elseif {$resname eq "HIS"  ||  $resname eq "HSD"} {
			set sc "CB CG"
		} elseif {$resname eq "ILE"} {
			set sc "CB CG1 CG2 CD CD1"
		} elseif {$resname eq "LYS"} {
			set sc "CB CG CD"
		} elseif {$resname eq "LEU"} {
			set sc "CB CG CD1 CD2"
		} elseif {$resname eq "MET"} {
			set sc "CB CG SD CE"
		} elseif {$resname eq "ASN"} {
			set sc "CB CG ND1 ND2 OD1 OD2 HD11 HD12 HD21 HD22"
		} elseif {$resname eq "PRO"} {
			set sc "CB CG CD"
		} elseif {$resname eq "GLN"} {
			set sc "CB CG CD OE1 OE2 NE1 NE2 HE11 HE12 HE21 HE22"
		} elseif {$resname eq "PRO"} {
			set sc "CB CG CD"
		} elseif {$resname eq "ARG"} {
			set sc "CB CG CD"
		} elseif {$resname eq "SER"} {
			set sc "CB OG HG"
		} elseif {$resname eq "THR"} {
			set sc "CB OG1 HG1 CG2"
		} elseif {$resname eq "VAL"} {
			set sc "CB CG1 CG2"
		} elseif {$resname eq "TRP"} {
			set sc "CB CG"
		} elseif {$resname eq "TYR"} {
			set sc "CB CG"
		} else {
			puts "ERROR: did not recognize residue $resname"
		}
		if {$sc ne ""} {

			set bbsel [atomselect "top" "protein and resid $res and name $bb"]
			set scsel [atomselect "top" "protein and resid $res and name $sc"]
			set casel [atomselect "top" "protein and resid $res and name CA"]
			set cbsel [atomselect "top" "protein and resid $res and name CB"]
	
			for {set f 0} {$f<$numframes} {incr f} {
				$bbsel frame $f
				$scsel frame $f
				$casel frame $f
				$cbsel frame $f
	
				set com_bb [measure center $bbsel weight mass] 
				set com_sc [measure center $scsel weight mass]
				$casel set x [lindex $com_bb 0]
				$casel set y [lindex $com_bb 1]
				$casel set z [lindex $com_bb 2]
	
				$cbsel set x [lindex $com_sc 0]
				$cbsel set y [lindex $com_sc 1]
				$cbsel set z [lindex $com_sc 2]

			}
			$bbsel delete
			$scsel delete
			$casel delete
			$cbsel delete
		}
	}
	# measure dihedrals
	puts "measuring SC BB BB SC dihedrals"
	# create empty string for GLY, ALA
	set empty ""
	for {set f 0} {$f<$numframes} {incr f} {
		lappend empty 0
	}
	for {set i 0} {$i < [expr $n-1]} {incr i} {
		
			set sel [atomselect "top" "(resid [lindex $lext $i] and name CB CA) or (resid [lindex $lext [expr $i+1]] and name CA CB)"]
					
			if {[$sel num] == 4} {
				set atoms [$sel get index]
				set a1 [lindex $atoms 1]
				set a2 [lindex $atoms 0]
				set a3 [lindex $atoms 2]
				set a4 [lindex $atoms 3]
				# puts "$a1 $a2 $a3 $a4"
				set phi [measure dihed "$a1 $a2 $a3 $a4" frame "all"]
			} else {
				set phi $empty
			}
			lappend dih $phi
			$sel delete
	}

	puts "\[ dihedrals ]"
	puts "; SC-BB-BB-SC dihedrals"

	# go through list and create dihedral potentials
	for {set i 0} {$i < [llength $dih]} {incr i} {
		# variables i, dih, typ, rsd, idx
		# find index of BB resid i
		set k [lsearch $rsd [expr $i+1]]
		# calculate dihedral only if this or next residue not equal to Ala, Gly
		if {[lindex $seq $i] ne "A" && [lindex $seq $i] ne "G" && [lindex $seq [expr 1+$i]] ne "A" && [lindex $seq [expr 1+$i]] ne "G"} {
		#	if {[lindex $ss $i] ne "C" && [lindex $ss [expr $i+1]] ne "C" } { 
				set a2 [lindex $idx $k]
				set a1 [lindex $idx [expr $k+1]]
				set a3 [lindex $idx [lsearch $rsd [expr $i+2]]]
				set a4 [lindex $idx [expr 1+ [lsearch $rsd [expr $i+2]]]]
				set angle [expr round([lindex $dih $i]) -180]
			
				# make sure that angle within [-180,180]
				if {$angle < -180} {
					set angle [expr $angle+360]		
				}
				if {$angle > 180} {
					set angle [expr $angle-360]
				}
				puts [format "%*d %*d %*d %*d      1 %*d %*d     1 ; [lindex $seq $i]([expr $i + 1])-[lindex $seq [expr 1+$i]]([expr $i + 2])" 5 $a1 5 $a2 5 $a3 5 $a4 6 $angle 5 $b]
		#	}
		}
	}
	puts "\[ angles ]"
	puts "; SC-BB-BB  and  BB-BB-SC"

	# go through list and create ReB angle potentials
	for {set i 0} {$i < [llength $dih]} {incr i} {
		# variables i, dih, typ, rsd, idx
		# find index of BB resid i
		set k [lsearch $rsd [expr $i+1]]
		# add angle potential only if this or next residue not equal to Ala, Gly
		if {[lindex $seq $i] ne "A" && [lindex $seq $i] ne "G" && [lindex $seq [expr 1+$i]] ne "A" && [lindex $seq [expr 1+$i]] ne "G"} {
		#		if {[lindex $ss $i] ne "C" && [lindex $ss [expr $i+1]] ne "C" } {

				set a2 [lindex $idx $k]
				set a1 [lindex $idx [expr $k+1]]
				set a3 [lindex $idx [lsearch $rsd [expr $i+2]]]
				set a4 [lindex $idx [expr 1+ [lsearch $rsd [expr $i+2]]]]
	
				puts [format "%*d %*d %*d %*d %*d %*d ; [lindex $seq $i]([expr $i + 1])-[lindex $seq $i]([expr $i + 1])-[lindex $seq [expr 1+$i]]([expr $i + 2]) SBB" 5 $a1 5 $a2 5 $a3 6 $n_reb 6 $theta1 5 $k1]
				puts [format "%*d %*d %*d %*d %*d %*d ; [lindex $seq $i]([expr $i + 1])-[lindex $seq [expr 1+$i]]([expr $i + 2])-[lindex $seq [expr 1+$i]]([expr $i + 2]) BBS" 5 $a2 5 $a3 5 $a4 6 $n_reb 6 $theta2 5 $k2]
		#	}
		}
	}

# delete pdb file
mol delete "top"

} 
