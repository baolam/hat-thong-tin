from typing import Dict

from .. import Graph
from .unit import Unit

class KnowledgeGraph(Graph):
    def __init__(self) -> None:
        super().__init__()

        # Cài đặt theo đối tượng là đơn vị chính
        self.units : Dict[str, Unit] = nn.ModuleDict()
    
    def add(self, u : Unit):
        _id = u.id.__str__()
        
        # Kiểm tra đã tồn tại
        if self.units.get(_id) is None:
            self.units[_id] = u

    def delete(self):
        pass

    def update(self):
        pass

    def query(self):
        pass