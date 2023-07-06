import sys
sys.path.append("./")

from src.graph.knowledge.test import notification_unit
# notification_unit.call()

from torch import rand
# x = rand((5, 64))

# dis = notification_unit.dis(x)
# print(dis)

# sim = notification_unit.sim(x)
# print(sim)
def new_infor(**kwargs):
    print("Dữ liệu thêm vào: ", kwargs)

notification_unit._add_call(new_infor)

x_run = rand((1, 64))
notification_unit.run(x_run, thres=10, trace=["Lâm", "dễ", "thương"])