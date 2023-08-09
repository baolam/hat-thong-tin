from torch import nn

from .node import MANAGE_NODES, Node, NODES
from .edge import MANAGE_EDGES, Edge, EDGES
from .utils.array import search, add
from .constant import IMPOSSIBLE, POSSIBLE


class Graph(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes : MANAGE_NODES = nn.ModuleDict()
        self.edges : MANAGE_EDGES = nn.ModuleDict()

    def add_node(self, node : Node):
        address = node.address()
        if self.nodes.get(address) == None:
            return IMPOSSIBLE
        self.nodes[address] = node
        return POSSIBLE

    def neighbor_nodes(self, node : Node):
        result = []
        for edge in node.edges():
            _edge = self.edges.get(edge)
            add(_edge.to_node(), result)
        return result

    def add_edge(self, from_node : Node, 
        to_node : Node, is_direct : bool = False ,**kwargs):
        # Có hai trường hợp thêm cạnh
        # Là cạnh có hướng: (thêm một theo thứ tự từ from --> to)
        # Là loại vô hướng: (thêm hai thứ từ từ from --> to và từ to --> from)
        # ------------------------------------------------------------------- #
        # THUẬT TOÁN:
        # B1. Lấy địa chỉ của from_node và to_node
        # B2. Do loại thứ tự from --> to luôn luôn xảy ra nên xử lí trước cho trường hợp này
        # B2.1. Node lưu trữ address của cạnh kết nối nên 
        # cần xây dựng address của các node kết nối từ node được chỉ định (hàm neighbor_nodes)
        # B2.2. Tiến hành xác định xem to_address đã tồn tại chưa với mục đích tránh lặp lại gây tốn
        # thời gian (hàm search(...) == -1)
        # B2.3. Nếu chưa thì tiến hành thêm vào quản lí (
        #   + Thêm address của cạnh vào node được chỉ định
        #   + Thêm address của cạnh vào danh sách cạnh của graph
        # )
        # B3. Hoạt động tương tự nếu là cạnh vô hướng :>
        assert isinstance(is_direct, bool)

        # Lấy địa chỉ của node        
        from_address = from_node.address()
        to_address = to_node.address()
        
        from_state = search(to_address, 
            self.neighbor_nodes(from_node)) == -1
        if not from_state:
            return IMPOSSIBLE
    
        edge1 = Edge(from_address, to_address, **kwargs)
        from_node.add_edge(edge1)
        self.edges[edge1.address()] = edge1

        to_state = search(from_address, 
            self.neighbor_nodes(to_node)) == -1
        if not is_direct:
            if not to_state:
                return IMPOSSIBLE
            
            edge2 = Edge(to_address, from_address, **kwargs)
            to_node.add_edge(edge2)
            self.edges[edge2.address()] = edge2

        return POSSIBLE


    def sub_graph(self, beginning_node : Node, depth : int):
        # Tạo subgraph từ graph chính với độ sâu (depth)
        # Xuất phát từ đỉnh được chỉ định
        
        # Nhiệm vụ: Quản lí độ cao của graph
        depth_tree = { [beginning_node.address()] : 0 }
        # Lưu trữ danh sách cạnh
        edges : EDGES = []

        # Xây dựng hàm tìm kiếm
        # Bản chất là thao tác duyệt
        def visit(node : Node):
            current_depth = depth_tree[node.address()]
            # Kiểm tra điều kiện dừng
            if current_depth == depth + 1:
                return
            
            # Lặp qua các cạnh kết nối với đỉnh hiện tại
            for edge_address in node.edges():
                edge : Edge = self.edges[edge_address]
        
                # Kiểm tra coi thử đã thêm chưa
                # Nếu tồn tại rồi thì skip
                neighbor_address = edge.to_node()
                if depth_tree.get(neighbor_address) is not None:
                    continue
                
                edges.append(edge)
                # Đỉnh kết nối với đỉnh hiện tại
                depth_tree[neighbor_address] = current_depth + 1
                visit(self.nodes[neighbor_address])
        
        # Tiến hành duyệt
        visit(beginning_node)
        nodes : NODES = [ self.nodes[node_address] 
            for node_address in depth_tree.keys() ]
        
        subgraph = SubGraph()
        subgraph.initalize(nodes, edges)
        return subgraph


class SubGraph(Graph):
    def __init__(self):
        super().__init__()

    def add_node(self):
        pass

    def add_edge(self):
        pass

    def initalize(self, nodes : NODES, edges : EDGES):
        self.nodes = nodes
        self.edges = edges