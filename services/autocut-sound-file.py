import sys
from moviepy.editor import AudioFileClip

mFile = input("Nhập file: ")
folder = -1
duration = -1

if mFile.split('.')[1] == "txt":
    print("Đọc file txt...")
    f = open(mFile, "r", encoding = "utf-8")
    
    lst = f.read().replace('\r', '') \
        .split('\n')
    f.close()
    
    mFile, folder, duration = lst
    duration = int(duration)

    print("Đường dẫn tới file âm thanh là: {}".format(mFile))
    print("Đường dẫn tới folder lưu trữ là: {}".format(folder))
    print("Khoảng thời gian một file là (đơn vị giây): {}".format(duration))

if mFile.split('.')[1] != "mp3":
    print("Định dạng file âm thanh không chuẩn")
    sys.exit(0)

audio = AudioFileClip(mFile)
print("Tổng thời gian của file bạn nhập là : {}s".format(audio.duration))

if folder == -1:
    folder = input("Nhập đường dẫn tới folder muốn lưu trữ: ")

import os
if os.path.exists(folder) == False:
    print("Tiến hành tạo folder tự động")
    os.makedirs(folder)

if duration == -1:
    duration = input("Nhập khoảng thời gian cắt 1 file (đơn vị s): ")
    duration = int(duration)

# Số lượng file con
import math
n = audio.duration / duration
num_file = math.ceil(n)

print("Tổng số file được tạo thành là: {}".format(num_file))
print("Tiến hành cắt file...")

_flag = 0
positions = []
for i in range(1, num_file):
    positions.append(
        (_flag, _flag + duration)
    )
    _flag += duration

# Tiến hành điều chỉnh duration file cuối
pos_st, pos_en = positions[-1]
pos_en = audio.duration
positions[-1] = (pos_st, pos_en)

print("Đang tiến hành tạo file...")

from tqdm import tqdm
for i in tqdm(range(len(positions))):
    pos_st, pos_en = positions[i]
    sub_audio = audio.subclip(pos_st, pos_en)

    file_name = '{}/{}_{}.mp3'.format(folder, pos_st, pos_en)
    sub_audio.write_audiofile(file_name)

print("HOÀN TẤT :>")