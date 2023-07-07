# Đồ thị thông tin đặc trưng
# Lưu trữ các tính chất suy luận
from typing import Dict
from torch import nn
from .. import Graph
from ...atom import Atom


class FeatureInformation(Graph):
    def __init__(self):
        super().__init__()
        self.atoms : Dict[str, Atom] = nn.ModuleDict()

    def add(self, atom : Atom):
        _id = atom.id.__str__()

        if self.atoms.get(_id) is None:
            self.atoms[_id] = atom