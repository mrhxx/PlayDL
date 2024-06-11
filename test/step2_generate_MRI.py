import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import preproc_utils
import skimage as ski
import numpy as np
import h5py

processed_path = '/home/xiaoxuan/HighResDatasets/Processed/LHQ1024_jpg'
#image_files = [file for file in os.listdir(processed_path) if file.endswith('.jpg')]
image_files = sorted([file for file in os.listdir(processed_path) if file.endswith('.jpg')])

num_images = len(image_files)

file_ext = '.jpg'
output_folder = '/home/xiaoxuan/HighResDatasets/Processed/LHQ1024_jpg_paired'

os.makedirs(output_folder, exist_ok=True)

max_count = num_images
image_res = [(256, 256), (256, 256), (256, 256), (512, 512), (512, 512), (512, 512), (512, 512)]
recon_res = [(128, 128), (192, 192), (256, 256), (320, 320), (384, 384), (448, 448), (512, 512)]
noise_lvl = [0.0010, 0.0015, 0.0025, 0.0030, 0.0045, 0.0060, 0.0100]

num_recon_pass = len(recon_res)

index = 0
for i in range(num_recon_pass):   
    recon_label = f'{recon_res[i][0]}x{recon_res[i][1]}'
    pass_output_folder = os.path.join(output_folder, f'{recon_label}')
    
    clean_image_path = os.path.join(pass_output_folder, 'clean')
    noisy_image_path = os.path.join(pass_output_folder, 'noisy')
    os.makedirs(clean_image_path, exist_ok=True)
    os.makedirs(noisy_image_path, exist_ok=True)
          
    count = 1
    for file in image_files:
        file_path = os.path.join(processed_path, file)
        _, file_name = os.path.split(file_path)
        file_prefix, file_ext = os.path.splitext(file_name)
        
        # Read raw image data.
        gray_data = ski.io.imread(file_path)
        
        # Downsample in image domain to get clean data.
        clean_data = preproc_utils.downsample_image(gray_data, image_res[i])
        clean_data = clean_data / np.linalg.norm(clean_data, 'fro')
        clean_data_max = clean_data.max()
    
        # Downsample in Fourier domain to get MRI data.
        mri_data = preproc_utils.downsample_image_by_fft(gray_data, recon_res[i], image_res[i])
        mri_data = np.abs(mri_data)
        mri_data = mri_data / np.linalg.norm(mri_data, 'fro')
        mri_data_max = mri_data.max()
    
        image_data_max = max(clean_data_max, mri_data_max)
        clean_data = clean_data / image_data_max
        mri_data = mri_data / image_data_max
    
        # Get noise variance based on MRI data.
        noise_var = noise_lvl[i] * np.random.rand()
        
        noisy_data = ski.util.random_noise(mri_data, mode='gaussian', var=noise_var)
        noisy_data_max = noisy_data.max()
        
        # Rescale images based on max across images.
        image_data_max = max(image_data_max, noisy_data_max)
    
        clean_image_data = clean_data / image_data_max
        noisy_image_data = noisy_data / image_data_max
        
        # TO DO: write mri_data, noisy_data to HDF5 files.
        h5_file_name = f'{index:0{7}d}' + '.h5'
        h5_file_path = os.path.join(output_folder, h5_file_name)
        with h5py.File(h5_file_path, 'w') as handle:
            clean_dataset = handle.create_dataset('clean', shape=(1, *image_res[i]), dtype=np.float32, chunks=True)
            noisy_dataset = handle.create_dataset('noisy', shape=(1, *image_res[i]), dtype=np.float32, chunks=True)
            
            clean_dataset[0] = clean_image_data.astype(np.float32)
            noisy_dataset[0] = noisy_image_data.astype(np.float32)
        index += 1
        
        # Save to images for viewing purpose.
        clean_image = (clean_image_data * 255).astype(np.uint8)
        clean_image_file_path = os.path.join(clean_image_path, file_prefix + file_ext)
        ski.io.imsave(clean_image_file_path, clean_image, quality=100)

        noisy_image = (noisy_image_data * 255).astype(np.uint8)
        noisy_image_file_path = os.path.join(noisy_image_path, file_prefix + file_ext)
        ski.io.imsave(noisy_image_file_path, noisy_image, quality=100)
        
        if count % 100 == 0:
            print(f'Pass {i + 1}/{num_recon_pass}, Processed: {count}/{num_images}')
        count += 1

        if count > max_count:
            print(f'Pass {i + 1}/{num_recon_pass}, Reached maximum file limit: {max_count}')
            break