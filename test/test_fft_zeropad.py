import numpy as np
import matplotlib.pyplot as plt

# Create an N by N image (e.g., N=8)
N = 8
image = np.random.rand(N, N)

# Perform FFT without padding
fft_image = np.fft.fft2(image)

# Perform FFT with zero-padding to M by M (e.g., M=16)
M = 16
fft_image_padded = np.fft.fft2(image, s=(M, M))

# Plot to compare the absolute values of the FFT results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(fft_image), cmap='gray')
plt.title('FFT without Padding')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.abs(fft_image_padded), cmap='gray')
plt.title('FFT with Zero-Padding')
plt.colorbar()

plt.show()
