from typing import Tuple
from torch import Tensor, nn
from torch import rand
from . import SingeCore


class WordUnit(SingeCore):
    def __init__(self, gamma: int, beta: int, hidden: int, word : str):
        super().__init__(gamma, beta, hidden)
        self.id = word

        self.embedding = nn.Parameter(rand(gamma), requires_grad = True)
        self.extract = nn.Linear(beta, hidden)
        self.activate1 = nn.ReLU()

        self.transfrom_embed = nn.Linear(gamma, hidden)
        self.activate2 = nn.ReLU()

        self.expression = nn.Linear(hidden, beta)

        # Chuẩn hóa sau khi tổng hợp embedding
        self.norm = nn.LayerNorm(hidden)
    
    def __transfrom_embed(self) -> Tensor:
        e = self.embedding
        e = self.transfrom_embed(e)
        e = self.activate2(e)
        return e

    def forward(self, x: Tensor) -> Tensor:
        x = self.extract(x)
        x = self.activate1(x)

        e = self.__transfrom_embed()
        
        z = x + e
        z = self.norm(z)

        z = self.expression(z)
        return z