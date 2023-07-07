from typing import Dict, Tuple
from ..atom import Atom
from . import Learner
from torch import nn, no_grad, inf
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torchmetrics.classification import BinaryAccuracy


class DataCollection():
    def __init__(self, x : Tensor, 
        atom : Atom, device : str):
        self.__lst = atom.relates
        self.x = x
        self.device = device

    def strengthen_data(self, atoms : Dict[str, Atom]):
        # Xây dựng dữ liệu huấn luyện cho vấn đề xác định tính chất
        lst = []
        for _relate in self.__lst:
            atoms[_relate]

    def dataset(self) -> Tuple[TensorDataset, bool]:
        # Dữ liệu huấn luyện x, exp, exist
        # Xây dựng bộ dữ liệu theo tiêu chí đó
        # Cùng kích thước
        if self.__lst.__len__() == 0:
            return TensorDataset(
                self.x.to(self.device), 
                self.x.to(self.device)
            ), False


class SingleLearner(Learner):
    def __init__(self, atom : Atom, device : str = "cpu", threshold : float = 0.6):
        super().__init__()
        self.atom = atom.to(device)
        self.device = device
        self.reconstruction_loss = nn.MSELoss()
        self.binary_loss = nn.BCELoss()
        self.accuracy = BinaryAccuracy(threshold).to(device)

    def build_collection(self, x : Tensor, device : str = "cpu") -> DataCollection:
        return DataCollection(x, self.atom, device)

    def run(self, train : DataLoader, val : DataLoader = None, epochs: int = 10, is_enough = True):
        hist = []
        for __ in range(1, epochs + 1):
            hist.append(
                self.__run_batch(train, val, is_enough)
            )
        return hist

    def __run_batch(self, train : DataLoader, val : DataLoader, is_enough : bool):
        train_loss = 0.
        
        for __, data in enumerate(train):
            if is_enough:
                x, y1, y2 = data
                y2 = Variable(y2)
            else:
                x, y1 = data

            x = Variable(x)
            y1 = Variable(y1)

            self.optimizer.zero_grad()
            
            exp, exist = self.atom.forward(x)
            # print(exp.size(), y1.size())
            
            rl = self.reconstruction_loss(exp, y1)
            le = 0.
            if is_enough:
                le = self.binary_loss(exist, y2)

            l = rl + le
            l.backward()
            self.optimizer.step()

            train_loss += l.item()
        

        train_loss /= len(train)
        
        if val is None:
            return train_loss, inf
        
        val_loss = 0.
        acc = 0.
        
        with no_grad():
            for __, data in enumerate(train):
                if is_enough:
                    x, y1, y2 = data
                    y2 = Variable(y2)
                else:
                    x, y1 = data

                x = Variable(x)
                y1 = Variable(y1)
                
                exp, exist = self.atom.forward(x)
                rl = self.reconstruction_loss(exp, y1)
                le = 0.
                if is_enough:
                    le = self.binary_loss(exist, y2)
                    acc += self.accuracy(exist, y2).item()

                l = rl + le
                val_loss += l.item()
        
        val_loss /= len(val)
        acc /= len(val)

        return train_loss, val_loss, acc