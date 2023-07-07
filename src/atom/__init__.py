from typing import List, Tuple
from uuid import uuid4, UUID
from torch import nn
from torch import Tensor


class Atom(nn.Module):
    def __init__(self):
        super().__init__()
        # Id của hạt
        self.id = uuid4()
        # Tập các biểu hiện của hạt (lưu trữ id)
        self.exps : List[UUID] = []
        # Tập các tính chất liên quan
        self.relates : List[UUID] = []


    def forward(self):
        pass

    def best_kunit(self):
        '''
            Description:
            Xác định biểu hiện phù hợp nhất trong tập các biểu hiện đã lưu
        '''
        pass

    def add_exp(self, unit_id : UUID):
        # Nên cài đặt theo thứ tự có sẵn
        _id = unit_id.__str__()

        for exp in self.exps:
            if _id == str(exp):
                return
        
        self.exps.append(unit_id)

    def add_relate(self, another : UUID):
        # Cài đặt mảng sắp xếp
        _id = another.__str__()

        for relate in self.relates:
            if _id == relate:
                return
        
        self.relates.append(another)
        
    def forward(self, x : Tensor) -> Tuple[Tensor, Tensor]:
        pass