from glob import glob
import os
from .gmx.rtp import read_rtp
from . import DATA_PATH


class ForceField(object):
    def __init__(self, directory):
        source_files = glob(os.path.join(directory, '*.rtp'))
        blocks = {}
        links = []
        for source in source_files:
            with open(source) as infile:
                file_blocks, file_links = read_rtp(infile)
            blocks.update(file_blocks)
            links.extend(file_links)

        self.name = os.path.basename(directory)
        self.blocks = blocks
        self.links = links
        self.reference_graphs = blocks
        self.modifications = []
        self.renamed_residues = {}


def find_force_fields(directory):
    """
    Find all the force fields in the given directory.

    A force field is defined as a directory that contains at least one RTP
    file. The name of the force field is the base name of the directory.
    """
    force_fields = {}
    directory = str(directory)  # Py<3.6 compliance
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if glob(os.path.join(path, '*.rtp')):
                force_fields[name] = ForceField(path)
    return force_fields


FORCE_FIELDS = find_force_fields(os.path.join(DATA_PATH, 'force_fields'))
