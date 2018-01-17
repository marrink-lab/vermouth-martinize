from .processor import Processor


def apply_posres(molecule, selector, force_constant):
    for key, node in molecule.nodes.items():
        if selector(node):
            parameters = [1, ] + [force_constant, ] * 3
            molecule.add_interaction('position_restraints', (key, ), parameters)
    return molecule


class ApplyPosres(Processor):
    def __init__(self, selector, force_constant):
        super().__init__()
        self.selector = selector
        self.force_constant = force_constant

    def run_molecule(self, molecule):
        return apply_posres(molecule, self.selector, self.force_constant)
