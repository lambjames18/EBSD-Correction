import os
import numpy as np
from skimage import io, transform
import imageio


#### USER INPUTS ####
lower_resolution = 2.5
higher_resolution = 1.3
higher_res_image_path = 'test_data/BSE.tif'
image_dtype = np.uint8
flip_image = False
#### END USER INPUTS ####

folder = os.path.dirname(higher_res_image_path)
basename = os.path.splitext(os.path.basename(higher_res_image_path))[0]
save_name = os.path.join(folder, basename + '_rescaled.tif')

downscale_factor = higher_resolution / lower_resolution
higher_res_image = io.imread(higher_res_image_path, as_gray=True)
higher_res_image = higher_res_image.astype(np.float32)
higher_res_image = higher_res_image / np.max(higher_res_image)
lower_res_image = transform.rescale(higher_res_image, downscale_factor, anti_aliasing=True)
lower_res_image = lower_res_image.astype(np.float32)
if image_dtype == np.uint8:
    lower_res_image = np.around(255 * ((lower_res_image - np.min(lower_res_image)) / (np.max(lower_res_image) - np.min(lower_res_image)))).astype(np.uint8)
elif image_dtype == np.uint16:
    lower_res_image = np.around(65535 * ((lower_res_image - np.min(lower_res_image)) / (np.max(lower_res_image) - np.min(lower_res_image)))).astype(np.uint16)
else:
    lower_res_image = lower_res_image.astype(image_dtype)

print('Original image shape: {}'.format(higher_res_image.shape))
print('Rescaled image shape: {}'.format(lower_res_image.shape))
print('Saving rescaled image to: {}'.format(save_name))
if flip_image:
    lower_res_image = np.flip(lower_res_image, axis=0)
imageio.imwrite(save_name, lower_res_image)
