import skimage as ski
import numpy as np

def calc_image_mass_center(img: np.array):

    # Obtain the dimensions of the image
    height, width = img.shape
    
    # Create coordinate grids for x and y
    y_indices, x_indices = np.indices((height, width))
    
    # Calculate the total mass (sum of all intensities)
    total_mass = np.sum(img)
    
    # Avoid division by zero in case of a completely black image
    if total_mass == 0:
        return (0, 0)
    
    # Compute the mass center coordinates
    x_center = round(np.sum(img * x_indices) / total_mass)
    y_center = round(np.sum(img * y_indices) / total_mass)
    
    return (x_center, y_center)

def crop_image_to_square(img: np.array):

    # First calculate image mass center
    height, width = img.shape
    x_center, y_center = calc_image_mass_center(img)

    square_size = min(height, width)
    offset_minus = round(square_size / 2)
    offset_plus = square_size - offset_minus - 1

    # Determine the bounds of the crop
    left = max(0, x_center - offset_minus)
    right = min(width, x_center + offset_plus)
    top = max(0, y_center - offset_minus)
    bottom = min(height, y_center + offset_plus)
    
    # Adjust the coordinates to maintain the square shape if necessary
    if (right - left) < square_size:
        if left == 0:  # Adjust right
            right = min(width, left + square_size)
        else:  # Adjust left
            left = max(0, right - square_size)
    
    if (bottom - top) < square_size:
        if top == 0:  # Adjust bottom
            bottom = min(height, top + square_size)
        else:  # Adjust top
            top = max(0, bottom - square_size)

    # Crop the image
    return img[top:bottom, left:right]

def calculate_2d_fft(img: np.array, dim = None):
    if dim is not None:
        height, width = img.shape
        if dim[0] < height or dim[1] < width:
            raise RuntimeError('Requested FFT has a smaller size than input data')
        
    return np.fft.fftshift(np.fft.fft2(img, dim))

def calculate_2d_ifft(ksp: np.array, dim = None):
    if dim is not None:
        height, width = ksp.shape
        if dim[0] < height or dim[1] < width:
            raise RuntimeError('Requested IFFT has a smaller size than input data')
    return np.fft.ifft2(np.fft.ifftshift(ksp), dim)

def downsample_image(img: np.array, dim):
    return ski.transform.resize(img, dim)

def downsample_image_by_fft(img: np.array, act_size, recon_size = None):
    if recon_size is None:
        recon_size = act_size
    
    ksp = calculate_2d_fft(img)
    recon_mask = get_central_kspace_mask(ksp.shape, recon_size)

    ksp = ksp[recon_mask.astype(bool)].reshape(recon_size)
    act_mask = get_central_kspace_mask(ksp.shape, act_size)

    ksp = ksp * act_mask
    return calculate_2d_ifft(ksp)

def get_central_kspace_mask(ksp_dim, mask_dim):

    ksp_dim = np.array(ksp_dim)
    mask_dim = np.array(mask_dim)

    mask = np.zeros(ksp_dim)
    half_size = np.ceil(mask_dim / 2)

    offset_minus = half_size
    offset_plus = mask_dim - offset_minus

    center_index = np.ceil(ksp_dim / 2)

    top = int(center_index[0] - offset_minus[0])
    bottom = int(center_index[0] + offset_plus[0])
    left = int(center_index[1] - offset_minus[1])
    right = int(center_index[1] + offset_plus[1])

    mask[top:bottom, left:right] = True

    return mask