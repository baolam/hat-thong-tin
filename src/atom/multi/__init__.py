from typing import List, Dict
from torch import Tensor
from torch import nn
from ..single import SingeCore
from .. import Atom


class MultiCores(Atom):
    def __init__(self, singles : List[str] = []):
        super().__init__()
        self.singles = singles

    def add_core(self, core : SingeCore):
        _id = core.id.__str__()

        for single in self.singles:
            if single == _id:
                return

        self.singles.append(_id)
    
    def forward(self, x : Tensor, atoms : Dict[str, SingeCore]) -> Tensor:
        _e = 0
        for single in self.singles:
            _single = atoms[single]
            _e += _single.forward(x, is_existed = False)
        _e /= len(self.singles)
        return _e