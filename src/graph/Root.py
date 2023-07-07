from torch import nn
from .feature_information import FeatureInformation
from .knowledge import KnowledgeGraph
from ..utils.Logger import Logger


class Root(nn.Module):
    def __init__(self) -> None:
        super().__init__()

