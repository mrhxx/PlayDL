import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import skimage as ski
import numpy as np

dataset_folder = '/home/xiaoxuan/HighResDatasets/LHQ1024_jpg'
image_files = [file for file in os.listdir(dataset_folder) if file.endswith('.jpg')]

num_images = len(image_files)

output_file_ext = '.jpg'
output_folder = '/home/xiaoxuan/HighResdatasets/Processed/LHQ1024_jpg'
os.makedirs(output_folder, exist_ok=True)

count = 1
for file in image_files:
    file_path = os.path.join(dataset_folder, file)
    _, file_name = os.path.split(file_path)
    file_prefix, file_ext = os.path.splitext(file_name)
    
    rgb_data = ski.io.imread(file_path)
    gray_data = ski.color.rgb2gray(rgb_data)
    gray_data = (gray_data / gray_data.max() * 255).astype(np.uint8)
    
    output_path = os.path.join(output_folder, file_prefix + output_file_ext)
    ski.io.imsave(output_path, gray_data, quality=100)
    
    if count % 100 == 0:
        print(f'Processed: {count}/{num_images}')
    count = count + 1
    
    