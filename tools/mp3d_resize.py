import os
import cv2
from tqdm import tqdm


def downsample_images(input_folder, output_folder, target_size):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".png"): 
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            downsampled_image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(output_path, downsampled_image)
    print('Done!')

data_path = '/home/ka/ka_iar/ka_ba9856/mmsegmentation/data/MP3DPANO/semantic'
for split in ['train', 'val', 'test']:
    input_folder = os.path.join(data_path, split)
    input_folder = input_folder + '_old'
    output_folder = os.path.join(data_path, split) 
    target_size = (2048, 1024) 
    print(f'Resize {split}:')
    downsample_images(input_folder, output_folder, target_size)
    print('Done!')