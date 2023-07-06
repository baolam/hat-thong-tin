from . import SingeCore
from ...graph.knowledge import KnowledgeGraph
from torch import nn
from torch import rand, Tensor
from torch import relu, sigmoid


class StandardUnit(SingeCore):
    def __init__(self, gamma: int, beta: int, hidden: int):
        super().__init__(gamma, beta, hidden)

        # Đại diện tính toán của đơn vị
        self.avatar = nn.Parameter(rand(gamma))
        self.avt_dim = nn.Linear(gamma, hidden)

        # Trích xuất đặc trưng
        self.extract = nn.Linear(beta, hidden)
        # Chuẩn hóa khi kết hợp đại diện với chiều biểu hiện
        self.norm = nn.LayerNorm(hidden)

        # Ma trận biểu hiện của đơn vị
        self.expression = nn.Linear(hidden, beta)

        # Xác định tính chất
        self.exist = nn.Linear(beta, 1)
        
    def forward(self, x : Tensor):
        a = self.avatar
        a = self.avt_dim(a)
        a = relu(a)

        x = self.extract(x)
        x = relu(x)

        x *= a
        x = self.norm(x)

        x = self.expression(x)
        e = self.exist(x)
        e = sigmoid(e)

        return x, e
    
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

