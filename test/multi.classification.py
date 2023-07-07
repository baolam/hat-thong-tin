import sys
sys.path.append("./")

from src.atom.multi.MultiClassification import MultiClassification
classify = MultiClassification(3, 64)

from src.atom.single.StandardUnit import StandardUnit
u1 = StandardUnit(32, 64, 16)
u2 = StandardUnit(32, 64, 8)
u3 = StandardUnit(32, 64, 12)

atoms = {
    str(u1.id) : u1,
    str(u2.id) : u2,
    str(u3.id) : u3
}

classify.add_core(u1)
classify.add_core(u2)
classify.add_core(u3)

from torch import rand
x = rand((5, 64))

y = classify.forward(x, atoms)
print(y)

l = y.sum()
print(l)
l.backward()