from torch import nn


# Lớp này xây dựng cho encode ngôn ngữ
class AttentionLayer(nn.Module):
    def __init__(self, beta : int, hidden : int):
        super().__init__()