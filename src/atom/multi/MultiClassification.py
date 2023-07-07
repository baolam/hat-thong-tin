from typing import Dict, List
from torch import Tensor, nn

from src.atom.single import SingeCore
from . import MultiCores


class MultiClassification(MultiCores):
    def __init__(self, classes : int, beta : int, singles: List[str] = []):
        super().__init__(singles)
        self.classify = nn.Linear(beta, classes)
        self.activate = nn.Softmax(dim=1)

        if classes == 2:
            self.activate = nn.Sigmoid()
    
    def forward(self, x: Tensor, atoms: Dict[str, SingeCore]) -> Tensor:
        e = super().forward(x, atoms)
        e = self.classify(e)
        e = self.activate(e)
        return e