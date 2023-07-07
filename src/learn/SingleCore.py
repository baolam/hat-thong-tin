from typing import Dict, Tuple
from torch import nn, no_grad
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryAccuracy
from ..atom.single import SingeCore
from . import Learner


class DataCollection():
    # Các trạng thái huấn luyện
    ADEQUATE_CODE = 0
    RECONSTRUCTION_CODE = 1
    EXIST_CODE = 2

    def __init__(self, x : Tensor, y : Tensor,
        atom : SingeCore, device : str):
        self.state = self.ADEQUATE_CODE
        self.__lst = atom.relates
        if len(self.__lst) == 0:
            self.state = self.EXIST_CODE

        self.x = x
        if y is None:
            self.state = self.RECONSTRUCTION_CODE
            self.y = x
        
        self.device = device

    def strengthen_data(self, atoms : Dict[str, SingeCore]):
        # Xây dựng dữ liệu huấn luyện cho vấn đề xác định tính chất
        lst = []
        for _relate in self.__lst:
            atoms[_relate]

    def dataset(self) -> Tuple[TensorDataset, int]:
        # Dữ liệu huấn luyện x, exp, exist
        # Xây dựng bộ dữ liệu theo tiêu chí đó
        # Cùng kích thước
        if self.__lst.__len__() == 0:
            return TensorDataset(
                self.x.to(self.device), 
                self.y.to(self.device)
            ), self.state


class SingleLearner(Learner):
    def __init__(self, atom : SingeCore, device : str = "cpu", threshold : float = 0.6):
        super().__init__()
        self.device = device
        self.atom = atom.to(device)
        # Các thông số huấn luyện
        self.reconstruction_loss = nn.MSELoss()
        self.binary_loss = nn.BCELoss()
        self.accuracy = BinaryAccuracy(threshold).to(device)

    def build_collection(self, x : Tensor, 
        device : str = "cpu") -> DataCollection:
        return DataCollection(x, self.atom, device)

    def run(self, train : DataLoader, val : DataLoader = None, epochs: int = 10, is_enough = True):
        hist = []
        for __ in range(1, epochs + 1):
            hist.append(
                self.__run_batch(train, val, is_enough)
            )
        return hist

    def __run_batch(self, train : DataLoader, val : DataLoader, 
        state_code : int):
        if DataCollection.ADEQUATE_CODE == state_code:
            return self.__adequate_form(train, val)
        return self.__missing_form(train, val, state_code)

    def __adequate_form(self, train : DataLoader, val : DataLoader):
        train_loss = 0.

        for x, y1, y2 in train:
            self.optimizer.zero_grad()

            y1_hat, y2_hat = self.atom.forward(x, is_existed = True)
            l = self.reconstruction_loss(y1_hat, y1) + self.binary_loss(y2_hat, y2)
            l.backward()

            self.optimizer.step()
            train_loss += l.item()
        
        train_loss /= len(train)

        if val is None:
            return train_loss
        
        val_loss = 0.
        val_acc = 0.
        with no_grad():
            for x, y1, y2 in val:
                y1_hat, y2_hat = self.atom.forward(x, is_existed = True)
                l = self.reconstruction_loss(y1_hat, y1) + self.binary_loss(y2_hat, y2)
                val_loss += l.item()
                val_acc += self.accuracy(y2_hat, y2).item()
        
        val_loss /= len(val)
        val_acc /= len(val)

        return train_loss, val_loss, val_acc

    def __missing_form(self, train : DataLoader, val : DataLoader, 
        state_code : int):
        train_loss = 0.

        mem = False # Tượng trưng cho reconstruct
        loss_function = self.reconstruction_loss
        if state_code == DataCollection.EXIST_CODE:
            mem = True
            loss_function = self.binary_loss

        for x, y in train:
            self.optimizer.zero_grad()

            res = self.atom.forward(x, is_existed = mem)

            # Xác định đữ liệu
            y_hat = res
            if mem:
                __, y_hat = res

            l = loss_function(y_hat, y)
            l.backward()

            self.optimizer.step()
            train_loss += l.item()
        
        train_loss /= len(train)
        if val is None:
            return train_loss

        val_loss = 0.
        val_acc = 0.
        with no_grad():
            for x, y in val:
                res = self.atom.forward(x, is_existed = mem)
                
                y_hat = res
                if mem:
                    __, y_hat = res
                    val_acc += self.accuracy(y_hat, y).item()

                l = loss_function(y_hat, y)
                val_loss += l.item()

        val_loss /= len(val)
        val_acc /= len(val)

        return train_loss, val_loss, val_acc