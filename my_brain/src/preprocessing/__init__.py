from typing import Dict, Any
from torch import nn, vstack
from ..constant import IMPOSSIBLE, POSSIBLE, TENSOR
from .preprocess_node import PreprocessNode


class ProcessingSpace(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes : Dict[str, PreprocessNode] = nn.ModuleList()
    
    def add(self, process : PreprocessNode):
        if self.nodes.get(process.address()) is None:
            self.nodes[process.address()] = process
            return POSSIBLE
        return IMPOSSIBLE
    
    def forward(self, x : Any) -> TENSOR:
        # Nên cài đặt chạy trên thread
        res = []
        for process in self.nodes:
            r = process.format(x)        
            res.append(r)
        
        res = vstack(res)
        return res.mean(dim=0)