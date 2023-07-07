from torch import nn
from torch import Tensor


# Cài đặt theo bài báo Patches is all you need?
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels : int, hidden : int, patches : int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, hidden, 
            kernel_size=patches, stride=patches)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden)

    def forward(self, x : Tensor) -> Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        
        return x
    
class DepthWiseSeparable(nn.Module):
    pass

class ExtractImageLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.depthwise = nn.Dep