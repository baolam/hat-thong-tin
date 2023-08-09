from typing import Callable, Any
from torch import nn
from uuid import uuid4


class Handler(nn.Module):
    def __init__(self, inp_dim : int, process_func : Callable):
        super().__init__()
        self.inp_dim = inp_dim
        self.process_func : Callable = process_func


# Ý tưởng:
# Có hai hình thức tạo nên đặc trưng cho đầu vào
# Tạo đặc trưng dựa vào lập trình trên dữ liệu. Kết thúc bằng hình thành vector đặc trưng
# Dựa vào mô hình tự tối ưu. Kết thúc bằng hình thành vector đặc trưng
class PreprocessNode(nn.Module):
    def __init__(self, handler : Handler, address : str = None):
        super().__init__()
        self.handler = handler
        self.__address = address
        if address is None:
            self.__address = uuid4().__str__()

    def address(self):
        return self.__address
    
    def initalize(self, handler : Handler):
        self.handler = handler

    def forward(self, x : Any):
        if callable(self.process_function) is not None:
            x = self.handler.process_func(x)
        x = self.handler(x)
        return x