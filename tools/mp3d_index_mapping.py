import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


mapping = {
    0: [0],     # void
    1: [1],     # wall
    2: [2],     # floor
    3: [3, 19], # chair
    4: [4],     # door
    5: [5],     # table
    6: [6, 24], # picture
    7: [7, 13, 33, 34], # furniture
    8: [8, 12, 14, 20, 22, 27, 28, 29, 30, 35, 36, 37, 38, 39, 40, 41], # objects
    9: [9, 32], # window
    10: [10],   # sofa
    11: [11],   # bed
    12: [15],   # sink
    13: [16],   # stairs
    14: [17],   # ceiling
    15: [18],   # toilet
    16: [21],   # mirror
    17: [23],   # shower
    18: [25],   # bathhub
    19: [26],   # counter
    20: [31],   # shelving
}


max_index = max(max(values) for values in mapping.values()) + 1
mapping_array = np.zeros(max_index, dtype=np.uint8)

for new_index, old_indices in mapping.items():
    for old_index in old_indices:
        mapping_array[old_index] = new_index

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic indices mapping')
    parser.add_argument('--data-path', default='data/MP3DPANO/semantic', help='MP3D datasets path')
    args = parser.parse_args()
    return args

def process_file(file_path):
    img = Image.open(file_path)
    img_array = np.array(img)
    img_array = mapping_array[img_array]

    new_img = Image.fromarray(img_array)
    new_img.save(file_path)

def main():
    args = parse_args()

    data_split = ['train', 'val', 'test']
    for split in data_split:
        data_path = osp.join(args.data_path, split)
        for file in tqdm(os.listdir(data_path), desc=f"Processing {split} set"):
            if file.endswith(".png"):
                file_path = osp.join(data_path, file)
                process_file(file_path)

    print("Done!")

if __name__ == '__main__':
    main()
