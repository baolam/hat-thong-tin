from typing import Dict
from ..atom import Atom
from . import Learner
from torch import nn, no_grad, inf
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

class DataCollection():
    def __init__(self, x : Tensor, 
        atom : Atom):
        self.__lst = atom.relates

    def strengthen_data(self, atoms : Dict[str, Atom]):
        # Xây dựng dữ liệu huấn luyện cho vấn đề xác định tính chất
        lst = []
        for _relate in self.__lst:
            atoms[_relate]

    def dataset(self) -> TensorDataset:
        # Dữ liệu huấn luyện x, exp, exist
        # Xây dựng bộ dữ liệu theo tiêu chí đó
        # Cùng kích thước
        pass


class AtomLearner(Learner):
    def __init__(self, atom : Atom, device : str = "cpu"):
        super().__init__()
        self.atom = atom.to(device)
        self.device = device
        self.reconstruction_loss = nn.MSELoss()
        self.binary_loss = nn.BCELoss()
    
    def build_collection(self, x : Tensor) -> DataCollection:
        return DataCollection(x, self.atom)

    def run(self, train : DataLoader, val : DataLoader = None, epochs: int = 10):
        hist = []
        for e in range(1, epochs + 1):
            hist.append(
                (e, self.__run_batch(train, val))
            )
    
    def __run_batch(self, train : DataLoader, val : DataLoader):
        train_loss = 0.
        
        for x, y in train:
            x = Variable(x.to(self.device))
            y = Variable(y.to(self.device))

            self.optimizer.zero_grad()
            
            exp, __ = self.atom.forward(x)
            l = self.reconstruction_loss(exp, y)
            
            l.backward()
            self.optimizer.step()

            train_loss += l.item()
        

        train_loss /= len(train)
        
        if val is None:
            return train_loss, inf
        
        val_loss = 0.
        with no_grad():
            for x, y in val:
                x = Variable(x.to(self.device))
                y = Variable(y.to(self.device))

                exp, __ = self.atom.forward(x)
                l = self.reconstruction_loss(exp, y)

                val_loss += l.item()
        
        val_loss /= len(val)
        return train_loss, val_loss