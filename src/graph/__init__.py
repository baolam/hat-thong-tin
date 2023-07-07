from torch import nn


class Graph(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def add(self):
        # Thêm đơn vị mới
        pass

    def get(self):
        # Lấy đơn vị
        pass

    def put(self):
        # Cập nhật đơn vị
        pass

    def delete(self):
        # Xóa bỏ đơn vị
        pass

    def dfs(self):
        # Thăm các đơn vị theo ý tưởng DFS
        pass

    def bfs(self):
        # Thăm các đơn vị theo ý tưởng BFS
        pass