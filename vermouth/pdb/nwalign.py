# -*- coding: utf-8 -*-
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

'''

Sequence alignment with the Needleman-Wunsch algorithm

Designed to check the SEQRES entry of an input PDB versus the residues present

Adapted from
https://github.com/zaneveld/full_spectrum_bioinformatics/blob/master/content/08_phylogenetic_trees/needleman_wunsch_alignment.ipynb

'''
import numpy as np
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))

ONE_LETTER_AA = {"G": "GLY",
                 "A": "ALA",
                 "V": "VAL",
                 "C": "CYS",
                 "P": "PRO",
                 "L": "LEU",
                 "I": "ILE",
                 "M": "MET",
                 "W": "TRP",
                 "F": "PHE",
                 "S": "SER",
                 "T": "THR",
                 "Y": "TYR",
                 "N": "ASN",
                 "Q": "GLN",
                 "K": "LYS",
                 "R": "ARG",
                 "H": "HIS",
                 "D": "ASP",
                 "E": "GLU",
                 }

THREE_LETTER_AA = {ONE_LETTER_AA[key]: key for key in ONE_LETTER_AA.keys()}

def OLA_codes(res_list):
    '''
    convert continuous string of three letter AA codes into list of single letters
    '''
    n = 3
    return ''.join([THREE_LETTER_AA[i] for i in res_list])

def traceback_alignment(traceback_array, seq1, seq2, up_arrow= 1,
                        left_arrow=2, up_left_arrow=3, stop=0):
    """Align seq1 and seq2 using the traceback matrix and return as two strings

    traceback_array -- a numpy array with arrow characters indicating the direction from
    which the best path to a given alignment position originated

    seq1 - a sequence represented as a string
    seq2 - a sequence represented as a string
    up_arrow - the unicode used for the up arrows (there are several arrow symbols in Unicode)
    left_arrow - the unicode used for the left arrows
    up_left_arrow - the unicode used for the diagonal arrows
    stop - the symbol used in the upper left to indicate the end of the alignment
    """

    row = len(seq1)
    col = len(seq2)
    arrow = traceback_array[row, col]
    aligned_seq1 = ""
    aligned_seq2 = ""
    alignment_indicator = np.zeros(len(seq2))
    while arrow != 0:
        arrow = traceback_array[row, col]
        if arrow == up_arrow:
            # We want to add the new indel onto the left
            # side of the growing aligned sequence
            aligned_seq2 = "-" + aligned_seq2
            aligned_seq1 = seq1[row - 1] + aligned_seq1
            # alignment_indicator = " " + alignment_indicator
            row -= 1

        elif arrow == up_left_arrow:
            # Note that we look up the row-1 and col-1 indexes
            # because there is an extra "-" character at the
            # start of each sequence
            seq1_character = seq1[row - 1]
            seq2_character = seq2[col - 1]
            aligned_seq1 = seq1[row - 1] + aligned_seq1
            aligned_seq2 = seq2[col - 1] + aligned_seq2
            if seq1_character == seq2_character:
                # alignment_indicator = "|" + alignment_indicator
                alignment_indicator[col-1] = 1
            # else:

                # alignment_indicator = " " + alignment_indicator
            row -= 1
            col -= 1

        elif arrow == left_arrow:
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = seq2[col - 1] + aligned_seq2
            # alignment_indicator = " " + alignment_indicator
            col -= 1

        elif arrow == stop:
            break
        else:
            msg = (f"Traceback array entry at {row},{col}: {arrow} is not recognized as an up arrow"
                   "({up_arrow}),left_arrow ({left_arrow}), up_left_arrow ({up_left_arrow}), or a stop ({stop}).")
            raise ValueError(msg)

    return aligned_seq1, alignment_indicator, aligned_seq2


def nw(seq1, seq2, match = 1, mismatch = -1, indel = -1):

    n_rows = len(seq1)+1
    n_columns = len(seq2)+1

    scoring_array = np.full([n_rows, n_columns], 0)
    traceback_array = np.full([n_rows, n_columns], np.nan)
    up_arrow = 1
    left_arrow = 2
    up_left_arrow = 3

    for row in range(n_rows):
        for col in range(n_columns):
            if row == 0 and col == 0:
                # We're in the upper right corner
                score = 0
                arrow = 0
            elif row == 0:
                # We're on the first row
                # but NOT in the corner

                # Look up the score of the previous cell (to the left) in the score array
                previous_score = scoring_array[row, col - 1]
                # add the gap penalty to it's score
                score = previous_score + indel
                arrow = left_arrow
            elif col == 0:
                # We're on the first column but not in the first row
                previous_score = scoring_array[row - 1, col]
                score = previous_score + indel
                arrow = up_arrow
            else:
                # Calculate the scores for coming from above,
                # from the left, (representing an insertion into seq1)
                cell_to_the_left = scoring_array[row, col - 1]
                from_left_score = cell_to_the_left + indel

                # or from above (representing an insertion into seq2)
                above_cell = scoring_array[row - 1, col]
                from_above_score = above_cell + indel

                # diagonal cell, representing a substitution (e.g. A --> T)
                diagonal_left_cell = scoring_array[row - 1, col - 1]

                if seq1[row - 1] == seq2[col - 1]:
                    diagonal_left_cell_score = diagonal_left_cell + match
                else:
                    diagonal_left_cell_score = diagonal_left_cell + mismatch

                # take the max score
                score = max([from_left_score, from_above_score, diagonal_left_cell_score])

                # make note of which cell was the max in the traceback array
                if score == from_left_score:
                    arrow = left_arrow
                elif score == from_above_score:
                    arrow = up_arrow
                elif score == diagonal_left_cell_score:
                    arrow = up_left_arrow

            traceback_array[row, col] = arrow
            scoring_array[row, col] = score

    return scoring_array, traceback_array

def res_matching(alignment_indicator):
    diff = np.diff(alignment_indicator)
    missing_starts = np.where(diff == -1)[0]+1
    missing_ends = np.where(diff == 1)[0]+1

    if len(missing_starts) == len(missing_ends):
        if len(missing_starts) == 0:
            if np.unique(missing_starts) == 0:
                LOGGER.warning("Complete discrepancy between SEQRES and residues present in pdb file.",
                               type='pdb-fatal')
                return None
            else:
                return None

        # in this case both termini are either present or absent
        if missing_starts[0] < missing_ends[0]:
            #then everything's ok
            return (list(missing_starts), list(missing_ends))
        else:
            #ie. both termini are missing
            return (list(np.insert(missing_starts, 0, 0)),
                    list(np.append(missing_ends, len(alignment_indicator))))
    elif len(missing_starts) < len(missing_ends):
        # in this case the N terminus is not present.
        return (list(np.insert(missing_starts, 0, 0)),
                list(missing_ends))
    elif len(missing_starts) > len(missing_ends):
        # in this case the C terminus is not present.
        return (list(missing_starts),
                list(np.append(missing_ends, len(alignment_indicator))))


def seqalign(pdb, seqres, chain):

    seq1_OLA = OLA_codes(pdb)
    seq2_OLA = OLA_codes(seqres)

    scoring_array, traceback_array = nw(seq1_OLA, seq2_OLA, 1, -1, -1)
    pdb_aligned, alignment_indicator, seqres_aligned = traceback_alignment(traceback_array, seq1_OLA, seq2_OLA)
    # print(f"PDB   : {pdb_aligned}\n        {''.join(list(alignment_indicator.astype(int).astype(str)))}\n"
    #       f"SEQRES: {seqres_aligned}")
    if len(np.unique(alignment_indicator)) > 0:
        alignment_check = res_matching(alignment_indicator)
        if alignment_check is not None:
            for i, j in zip(alignment_check[0], alignment_check[1]):
                LOGGER.warning("SEQRES data suggests residues {}:{} in chain {} are missing",
                               i+1, j+1, chain,
                               type="pdb-alternate")