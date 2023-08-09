from torch import rand, nn
from . import Node


class WordNode(Node):
    def __init__(self, word : str, dim : int):
        super().__init__()
        assert isinstance(word, str)
        assert isinstance(dim, int)

        self._update_address(word)
        # Embedding của từ
        self.__embed = nn.Parameter(rand(dim))

    def role(self):
        return self.__embed