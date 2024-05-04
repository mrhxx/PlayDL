import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import preproc_utils
import skimage as ski
import numpy as np

test_image = '~/OneDrive/Desktop/0364.png'
image_data = ski.color.rgb2gray(ski.io.imread(test_image))

image_size = image_data.shape
x_center, y_center = preproc_utils.calc_image_mass_center(image_data)

print(f'Image Size: {image_size}')
print(f'Mass Center: X = {x_center}, Y = {y_center}')

image_cropped = preproc_utils.crop_image_to_square(image_data)
print(f'Cropped Image Size: {image_cropped.shape}')

ski.io.imshow(image_data)
ski.io.show()

ski.io.imshow(image_cropped)
ski.io.show()

ksp_data = preproc_utils.calculate_2d_fft(image_cropped, (2048, 2048))
print(f'DFT of size:: {ksp_data.shape}')
ski.io.imshow(np.log(np.abs(ksp_data)), cmap='gray')
ski.io.show()

img_ifft = preproc_utils.calculate_2d_ifft(ksp_data)
print(f'IDFT of size:: {img_ifft.shape}')
ski.io.imshow(np.abs(img_ifft), cmap='gray')
ski.io.show()

image_lowres = preproc_utils.downsample_image(image_cropped, (512, 512))
print(f'Imaged resized to: {image_lowres.shape}')
ski.io.imshow(image_lowres, cmap='gray')
ski.io.show()

image_mri = preproc_utils.downsample_image_by_fft(image_cropped, (256, 256), (512, 512))
print(f'MRI reconstructed size:: {image_mri.shape}')
ski.io.imshow(np.abs(image_mri), cmap='gray')
ski.io.show()

image_lowres = image_lowres / np.linalg.norm(image_lowres, 'fro')
image_mri = image_mri / np.linalg.norm(image_mri, 'fro')

image_diff = np.abs(np.abs(image_lowres) - np.abs(image_mri))
ski.io.imshow(np.abs(image_diff), cmap='gray')
ski.io.show()