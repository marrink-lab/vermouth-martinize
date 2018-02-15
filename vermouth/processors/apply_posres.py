from .processor import Processor


def apply_posres(molecule, selector, force_constant, functype=1):
    for key, node in molecule.nodes.items():
        if selector(node):
            parameters = [functype, ] + [force_constant, ] * 3
            molecule.add_interaction('position_restraints', (key, ), parameters)
    return molecule


class ApplyPosres(Processor):
    def __init__(self, selector, force_constant, functype=1):
        super().__init__()
        self.selector = selector
        self.force_constant = force_constant
        self.functype = functype

    def run_molecule(self, molecule):
        return apply_posres(
            molecule,
            self.selector,
            self.force_constant,
            functype=self.functype
        )
