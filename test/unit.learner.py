import sys
sys.path.append("./")

from src.atom.single.StandardUnit import StandardUnit
unit = StandardUnit(gamma=64, beta=128, hidden=16)

from src.learn.SingleCore import SingleLearner
device = "cuda:0"
learner = SingleLearner(unit, device = device)

from torch.autograd import set_detect_anomaly
set_detect_anomaly(True)

from torch import optim
learner.set_optim(optim.Adam, lr = 0.01)

from torch.utils.data import DataLoader
from torch import rand

x = rand((64, 128))
dataset, is_enough = learner.build_collection(x, device=device).dataset()
loader = DataLoader(dataset, 16)

hist = learner.run(loader, is_enough=is_enough)
print(hist)
# exp, __ = unit.forward(x)
# r = exp.sum()
# # print(r)
# r.backward()