from typing import List
# Phục vụ cho thao tác trạng thái
from ..constant import IMPOSSIBLE, POSSIBLE

# Độ phức tạp các thao tác
# Chèn: O(n)
# Xóa: O(n)
# Tìm kiếm: O(logn)

def search(val : str, array : List[str]) -> int:
    # Cài đặt tìm kiếm trên mảng đã sắp xếp
    # Trả về vị trí trong mảng (-1 nghĩa là ko tìm thấy)
    l = 0
    r = len(array) - 1
    # Cài đặt tìm kiếm nhị phân
    while l <= r:
        m = (l + r) // 2
        # m = int(m)
        if val == array[m]:
            return m
        if array[m] < val:
            l = m + 1
        else:
            r = m - 1
    return -1

def add(val : str, array : List[str]):
    # Đặt giả thiết mảng được sắp xếp theo thứ tự tăng dần
    # Giả thiết không xảy ra trùng lặp
    # Không cần trả về giá trị
    # Can thiệp trực tiếp trên mảng
    if search(val, array) != -1:
        return IMPOSSIBLE
    # Tìm vị trí chèn
    pos = 0
    while pos <= len(array) - 1 \
        and val > array[pos]:
        pos += 1
    if pos == len(array):
        array.append(val)
    else:
        array.insert(pos, val)
    return POSSIBLE

def remove(val : str, array : List[str]):
    # Xóa phần tử trong mảng đã sắp xếp
    pos = search(val, array)
    if pos == -1:
        return IMPOSSIBLE
    array.pop(pos)
    return POSSIBLE