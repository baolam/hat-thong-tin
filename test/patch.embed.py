import sys
sys.path.append("./")

from src.layers.ExtractImageLayer import PatchEmbedding
emb = PatchEmbedding(3, 1, 16)

from torch import rand
x = rand((1, 3, 320, 320))

y = emb(x)
print(y.shape)