from . import SingeCore


class ImageUnit(SingeCore):
    def __init__(self, gamma: int, beta: int, hidden: int):
        super().__init__(gamma, beta, hidden)