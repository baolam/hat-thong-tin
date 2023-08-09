from typing import Callable
from uuid import uuid4
from torch import nn


class Edge(nn.Module):
    def __init__(self, from_node : str, to_node : str, 
        name_edge : str = None, 
        callback : Callable = None):
        super().__init__()

        assert isinstance(from_node, str)
        assert isinstance(to_node, str)
        
        self.__address = uuid4().__str__()
        self.__from_node = from_node
        self.__to_node = to_node
        self.__name_edge = name_edge

        # Hàm găn liền khi cạnh được kích hoạt
        self.callback = callback

    def from_node(self):
        return self.__from_node
    
    def to_node(self):
        return self.__to_node
    
    def name(self):
        return self.__name_edge
    
    def address(self):
        return self.__address
    
    def call(self, *args, **kwargs):
        # Cài đặt gọi callback khi thỏa một số điều kiện
        if callable(self.callback) is None:
            return
        self.callback(*args, **kwargs, 
            type=self.__type)

from typing import Dict, List
MANAGE_EDGES = Dict[str, Edge]
EDGES = List[Edge]