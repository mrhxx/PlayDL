import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import preproc_utils
import skimage as ski
import numpy as np

processed_path = '/home/xiaoxuan/HighResdatasets/Processed/LHQ1024_jpg'
#image_files = [file for file in os.listdir(processed_path) if file.endswith('.jpg')]
image_files = sorted([file for file in os.listdir(processed_path) if file.endswith('.jpg')])

num_images = len(image_files)

file_ext = '.jpg'
output_folder = '/home/xiaoxuan/HighResdatasets/Processed/LHQ1024_jpg_paired'

os.makedirs(output_folder, exist_ok=True)

image_res = tuple((512, 512))
recon_res = [(256, 256), (320, 320), (384, 384), (448, 448), (512, 512)]

clean_image_path = os.path.join(output_folder, 'clean')
noisy_image_path = os.path.join(output_folder, 'noisy')

os.makedirs(clean_image_path, exist_ok=True)
os.makedirs(noisy_image_path, exist_ok=True)

count = 1
max_count = 10000

for file in image_files:
    file_path = os.path.join(processed_path, file)
    _, file_name = os.path.split(file_path)
    file_prefix, file_ext = os.path.splitext(file_name)
    
    # Read raw image data.
    gray_data = ski.io.imread(file_path)
    
    # Downsample in image domain to get clean data.
    clean_data = preproc_utils.downsample_image(gray_data, image_res)
    clean_data = clean_data / np.linalg.norm(clean_data, 'fro')
    clean_data_max = clean_data.max()
    
    # Fourier domain downsampling.
    res_index = np.random.randint(len(recon_res))
    
    mri_data = preproc_utils.downsample_image_by_fft(gray_data, recon_res[res_index], image_res)
    mri_data = np.abs(mri_data)
    mri_data = mri_data / np.linalg.norm(mri_data, 'fro')
    mri_data_max = mri_data.max()
    
    image_data_max = max(clean_data_max, mri_data_max)
    clean_data = clean_data / image_data_max
    mri_data = mri_data / image_data_max
    
    # Get noise variance based on MRI data.
    noise_var = np.random.rand()*0.0025
    
    noisy_data = ski.util.random_noise(mri_data, mode='gaussian', var=noise_var)
    noisy_data_max = noisy_data.max()
    
    # Rescale images based on max across images.
    image_data_max = max(image_data_max, noisy_data_max)
    
    # TO DO: write mri_data, noisy_data to HDF5 files.
    
    # Save to images for viewing purpose.
    clean_image = (clean_data / image_data_max * 255).astype(np.uint8)
    clean_image_file_path = os.path.join(clean_image_path, file_prefix + file_ext)
    ski.io.imsave(clean_image_file_path, clean_image, quality=100)
    
    file_res = recon_res[res_index]
    
    noisy_image = (noisy_data / image_data_max * 255).astype(np.uint8)
    noisy_image_file_path = os.path.join(noisy_image_path, file_prefix + f'_{file_res[0]}x{file_res[1]}' + file_ext)
    ski.io.imsave(noisy_image_file_path, noisy_image, quality=100)
    
    if count % 100 == 0:
        print(f'Processed: {count}/{num_images}')
    count = count + 1
    
    if count > max_count:
        print(f'Reached maximum file limit: {max_count}')
        break