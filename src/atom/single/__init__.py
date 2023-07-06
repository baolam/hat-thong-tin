from .. import Atom


class SingeCore(Atom):
    def __init__(self, gamma : int, beta : int, hidden : int):
        super().__init__()

        assert isinstance(gamma, int)
        assert isinstance(beta, int)
        assert isinstance(hidden, int)

        assert gamma == beta, "Số chiều biểu diễn và số chiều đại diện phải khác nhau"
        self.gamma = gamma
        self.hidden = hidden
        self.beta = beta
    
    def total_params(self):
        print("Xây dựng hàm này :>")

    def learning_mode(self, mode : str):
        pass