# Đơn vị cơ bản gồm lưu trữ:
# Đại diện (vector) ứng với đơn vị
# Ngôn ngữ biểu thị (content)

from torch import nn
from torch import sqrt
from torch import rand, Tensor
class _Avatar(nn.Module):
    def __init__(self, avt_dim : int, **kwargs):
        super().__init__(**kwargs)
        # Xác định thiết bị hoạt động
        device = kwargs.get("device")
        if device == None:
            device = "cpu"
        self.code = nn.Parameter(rand((avt_dim))).to(device)

    def sim(self, x : Tensor) -> Tensor:
        # Tính toán độ tương đồng
        # Số chiều đầu vào : (N, avt_dim)
        # Số chiều của code : (avt_dim, 1)
        # Số chiều đầu ra : (N, 1)
        return x @ self.code.unsqueeze(1)

    def dis(self, x : Tensor) -> Tensor:
        # Tính khoảng cách
        # Số chiều đầu vào : (N, avt_dim)
        # Số chiều của code : (avt_dim, 1)
        # Số chiều đầu ra : (N)
        x -= self.code
        x = x ** 2
        x = x.sum(dim = 1)
        x = sqrt(x)
        return x
    
    def _device(self, dev):
        # Chạy trên thiết bị được chỉ định
        self.code = self.code.to(dev)
    

from typing import Dict, List
from uuid import uuid4, UUID
from ..KnowledgeRelation import Edge

class Unit(nn.Module):
    def __init__(self, content, avt_dim : int ,**kwargs):
        super().__init__(**kwargs)
        # Id của đơn vị
        self.id = uuid4()
        self.content = content
        self.avatar = _Avatar(avt_dim, **kwargs)

        # Nếu được kích hoạt
        # Gọi các hành động ứng với
        self.__callbacks = []
        # Lưu trữ quan hệ với các đơn vị khác
        self.relates : Dict[str, List[Edge]] = {}

    def _add_call(self, f):
        if not callable(f):
            return
        self.__callbacks.append(f)
    
    def _call(self, **kwargs):
        for function in self.__callbacks:
            function(**kwargs)

    def _add_relate(self, _type : str, _id : UUID) -> Edge:
        # Đây là mối quan hệ hai chiều
        edge = Edge(self.id, _type, _id)

        # Kiểm tra xem id đã tồn tại chưa
        _edges = self.relates.get(_type)
        if _edges is None:
            self.relates[_type] = edge
        else:
            flag = True

            # Lặp qua các cạnh và kiểm tra
            for _edge in _edges:
                if _edge._to == _id:
                    flag = False
                    break
            
            if flag:
                self.relates[_type].append(edge)
        
        return edge

    def sim(self, x : Tensor):
        return self.avatar.sim(x)
    
    def dis(self, x : Tensor):
        return self.avatar.dis(x)
    
    def run(self, x : Tensor, thres : float, **kwargs):
        # Chỉ cho duy nhất một điểm dữ liệu
        if x.size()[0] != 1:
            return
        dis = self.dis(x).item()
        if dis <= thres:
            self._call(**kwargs)