POWERPOINT_FOLDER = "F:\subjects\eng\listening\my_name_is_john"
SOURCE_FOLDER = "F:\subjects\eng\listening\children"

NUM_FILE_PER_SLIDE = 5 # Số file âm thanh mỗi slide

import os
folders = []
for item in os.listdir(SOURCE_FOLDER):
    _folder = f'{SOURCE_FOLDER}/{item}'
    isDirectory = os.path.isdir(_folder)
    if isDirectory:
        folders.append(_folder)

def sort_f(folder):
    f = folder.split('/')[1].split('_')[0]
    f = int(f)
    # print(folder.split('/')[1].split('_')[0])
    return f

folders.sort(key=sort_f)

import math
print(f'Tổng số file là: {len(folders)}')
SLIDES = math.ceil(len(folders) / NUM_FILE_PER_SLIDE)
print(f'Tổng số slide có thể có là: {SLIDES}')

import shutil
# Tiến hành group lại file
slides = []
# Số chỉ số bắt đầu
idx = 0
for __ in range(SLIDES):
    slide_folder = os.path.join(POWERPOINT_FOLDER, f'slide_{__ + 1}')
    # Kiểm tra nếu folder chưa tồn tại
    if os.path.exists(slide_folder) == False:
        os.makedirs(slide_folder)
    
    text = []
    for n in range(NUM_FILE_PER_SLIDE):
        i = idx + n
        if i == len(folders):
            break
        f = open(folders[i] + '/content.txt', "r", encoding="utf-8") 
        text.append(f.read().replace('\r\n', '') + '\n')
        f.close()

        # Xử lí với file âm thanh
        sound = folders[i] + '/content.mp3'
        shutil.copyfile(sound, slide_folder + f'/{i + 1}.mp3')
    
    idx += NUM_FILE_PER_SLIDE
    with open(slide_folder + '/content.txt', "w", encoding = "utf-8") as f:
        f.writelines(text)

    slides.append(slide_folder)

from moviepy.editor import AudioFileClip, concatenate_audioclips

import time
print("Đợi chờ là hạnh phúc")
time.sleep(2)

MAX_POWER_FILES = 7
POWER_FILES = math.ceil(len(slides) / MAX_POWER_FILES)
print(f'Tổng số file powerpoint dự kiến là: {POWER_FILES}')
powerpoints = []
idx = 0
for __ in range(POWER_FILES):
    power_folder = os.path.join(POWERPOINT_FOLDER, f'powerpoint_{__ + 1}')
    if os.path.exists(power_folder) == False:
        os.makedirs(power_folder)
    for n in range(MAX_POWER_FILES):
        i = idx + n
        if i == len(slides):
            break
        
        name = slides[i].split('\\')[-1]
        slide_path = power_folder + f'/{name}'
        
        shutil.move(slides[i], slide_path)
   
    idx += MAX_POWER_FILES
    powerpoints.append(power_folder)

print("Tiến hành tạo file nghe.")
for folder in powerpoints:
    audioclips = []
    for item in os.listdir(folder):
        _path = f'{folder}/{item}'
        if os.path.isdir(_path):
            for file in os.listdir(_path):
                if file.split('.')[1] == "mp3":
                    audioclips.append(AudioFileClip(_path + f'/{file}')
                )
    final_clip = concatenate_audioclips(audioclips)
    final_clip.write_audiofile(folder + '/listening_file.mp3')

print("HOÀN THÀNH :>")