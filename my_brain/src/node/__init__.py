from torch import nn
from torch import vstack
from uuid import uuid4
from ..constant import TENSOR
from ..edge import Edge, EDGES
from ..utils.array import add, remove


class Node(nn.Module):
    def __init__(self):
        super().__init__()

        self.__address = uuid4().__str__()
        # Tập các đỉnh kết nối với đỉnh này
        self.__edges : EDGES  = nn.ModuleList()
    
    def edges(self) -> EDGES:
        return self.__edges

    # -------------------------------------------- #
    # Quản lí dựa vào address của cạnh
    def add_edge(self, edge : Edge):
        return add(
            edge.address(), 
            self.__edges
        )

    def remove_edge(self, edge : Edge):
        return remove(
            edge.address(), 
            self.__edges
        )
    # -------------------------------------------- #

    def address(self):
        return self.__address
    
    def _update_address(self, address):
        self.__address = address
    
    def role(self) -> TENSOR:
        # Lấy vai trò của đỉnh
        pass

    def forward(self, x : TENSOR) -> TENSOR:
        # Tính toán ở tại đỉnh
        pass

    def forward_neighbor(self, x : TENSOR, node_dict, edge_dict,
        is_forward : bool = False) -> TENSOR:
        # Kết quả trả về có dạng: (M, N)
        # Với M là số neighbor, N là số điểm dữ liệu
        if is_forward == False:
            x = self.forward(x)

        # Lưu trữ kết quả tính toán
        res = []
        for edge in self.__edges:
            node = node_dict[edge_dict[edge]]

            # Sử dụng tích vô hướng để xác định tương quan
            out = x @ node.role()
            res.append(out)
        
        # Kết quả tính toán
        res = vstack(res)
        # Sửa lại chiều --> (M, N)
        res = res.reshape(
            len(self.__edges)
        )
        return res
    
from typing import Dict, List
MANAGE_NODES = Dict[str, Node]
NODES = List[Node]
