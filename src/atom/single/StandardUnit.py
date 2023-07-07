from . import SingeCore
from ...graph.knowledge import KnowledgeGraph
from torch import nn
from torch import rand, Tensor
from torch import relu, sigmoid


class StandardUnit(SingeCore):
    def __init__(self, gamma: int, beta: int, hidden: int):
        super().__init__(gamma, beta, hidden)

        # Đại diện tính toán của đơn vị
        self.avatar = nn.Parameter(rand(gamma), requires_grad=True)
        self.avt = nn.Linear(gamma, hidden)
        self.act1 = nn.ReLU()

        # Trích xuất đặc trưng
        self.extract = nn.Linear(beta, hidden)
        self.act2 = nn.ReLU()
        # Chuẩn hóa khi kết hợp đại diện với chiều biểu hiện
        self.norm = nn.LayerNorm(hidden)

        # Ma trận biểu hiện của đơn vị
        self.expression = nn.Linear(hidden, beta)

        # Xác định tính chất
        self.exist = nn.Linear(beta, 1)
    
    def avt_run(self) -> Tensor:
        a = self.avatar
        a = self.avt(a)
        a = self.act1(a)

        return a

    def forward(self, x : Tensor, is_existed : bool = True):
        z = self.extract(x)
        z = self.act2(z)

        a = self.avt_run()
        
        # Tránh phép toán +=
        k = z + a
        k = self.norm(k)
        k = self.expression(k)
        
        if is_existed:
            # Xác định tính chất
            e = self.exist(k)
            e = sigmoid(e)

            return k, e

        return k
    
    def best_kunit(self, x : Tensor, knowledge : KnowledgeGraph, threshold : float = 0.5):
        if x.size()[0] != 1:
            return
        exp, exist = self.forward(x)
        exist = exist.item()
        
        if exist >= threshold:
            _units = []
            for e in self.exps:
                _units.append(knowledge.units[str(e)])
    
    def learning_mode(self, mode: str):
        pass

