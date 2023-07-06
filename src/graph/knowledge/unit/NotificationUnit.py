from torch import Tensor
from . import Unit

class NotificationUnit(Unit):
    def __init__(self, content : str, avt_dim: int, **kwargs):
        # Đơn vị thông báo là hiển thị content (là ngôn ngữ)
        assert isinstance(content, str)
        
        super().__init__(content, avt_dim, **kwargs)
        
    def call(self):
        self._call(content=self.content)

    def run(self, x: Tensor, thres: float = 0.5, **kwargs):
        return super().run(x, thres, content=self.content ,**kwargs)