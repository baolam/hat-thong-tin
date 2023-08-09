from constant import TENSOR
from . import Node
from ..edge import Edge


class LinkingNode(Node):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: TENSOR) -> TENSOR:
        return x
    
    def forward_neighbor(self, x: TENSOR, node_dict, 
        edge_dict, is_forward: bool = False) -> TENSOR:
        return super().forward_neighbor(x, node_dict, edge_dict, is_forward)
        