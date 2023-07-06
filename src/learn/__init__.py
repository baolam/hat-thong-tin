from torch import optim

class Learner:
    def __init__(self):
        self.optimizer : optim.Optimizer = None

    def run(self, epochs : int):
        pass

    def run_batch(self):
        pass

    def set_optim(self, optimizer : optim.Optimizer, **kwargs):
        if issubclass(optimizer, optim.Optimizer) == False:
            return
        self.optimizer = optimizer(self.atom.parameters(), **kwargs)

    def learning_mode(self, mode : str):
        pass