from typing import List
from .MultiClassification import MultiClassification


class MultiStandard(MultiClassification):
    def __init__(self, beta: int, singles: List[str] = ...):
        super().__init__(2, beta, singles)