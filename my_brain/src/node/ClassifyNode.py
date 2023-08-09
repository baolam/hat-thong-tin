from torch import nn, rand

from constant import TENSOR
from . import Node


class ClassifyNode(Node):
    def __init__(self, num_classes : int ,dim : int):
        super().__init__()
        assert isinstance(num_classes, int)
        assert isinstance(dim, int)

        self.__avatar = nn.Parameter(rand(dim))

    def role(self) -> TENSOR:
        return self.__avatar