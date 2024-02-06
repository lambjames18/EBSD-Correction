import numpy as np
from skimage import transform
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import timeit

# Define the function to be tested
def skimage_upscale_aa():
    im = np.random.rand(1000, 1000)
    new_im = transform.resize(im, (2000, 2000), anti_aliasing=True)
    return

def skimage_upscale_no_aa():
    im = np.random.rand(1000, 1000)
    new_im = transform.resize(im, (2000, 2000), anti_aliasing=False)
    return

def numpy_upscale():
    im = np.random.rand(1000, 1000)
    new_im = np.repeat(np.repeat(im, 2, axis=0), 2, axis=1)
    return

def skimage_downscale_aa():
    im = np.random.rand(1000, 1000)
    new_im = transform.resize(im, (500, 500), anti_aliasing=True)
    return

def skimage_downscale_no_aa():
    im = np.random.rand(1000, 1000)
    new_im = transform.resize(im, (500, 500), anti_aliasing=False)
    return

def numpy_downscale():
    im = np.random.rand(1000, 1000)
    new_im = im[::2, ::2]
    return

def torch_upscale():
    im = torch.tensor(np.random.rand(1000, 1000)).unsqueeze(0).unsqueeze(0)
    new_im = resize(im, (2000, 2000), InterpolationMode.NEAREST)
    new_im = np.squeeze(new_im.detach().numpy())
    return

def torch_downscale():
    im = torch.tensor(np.random.rand(1000, 1000)).unsqueeze(0).unsqueeze(0)
    new_im = resize(im, (500, 500), InterpolationMode.NEAREST)
    new_im = np.squeeze(new_im.detach().numpy())
    return

# Run the tests
skimage_upscale_aa_time = timeit.timeit(skimage_upscale_aa, number=100)
skimage_upscale_no_aa_time = timeit.timeit(skimage_upscale_no_aa, number=100)
numpy_upscale_time = timeit.timeit(numpy_upscale, number=100)
torch_upscale_time = timeit.timeit(torch_upscale, number=100)
skimage_downscale_aa_time = timeit.timeit(skimage_upscale_aa, number=100)
skimage_downscale_no_aa_time = timeit.timeit(skimage_upscale_no_aa, number=100)
numpy_downscale_time = timeit.timeit(numpy_downscale, number=100)
torch_downscale_time = timeit.timeit(torch_downscale, number=100)

# Print the results
print("Upscale times:")
print("skimage:", skimage_upscale_aa_time)
print("skimage:", skimage_upscale_no_aa_time)
print("numpy:", numpy_upscale_time)
print("torch:", torch_upscale_time)
print("Downscale times:")
print("skimage:", skimage_downscale_aa_time)
print("skimage:", skimage_downscale_no_aa_time)
print("numpy:", numpy_downscale_time)
print("torch:", torch_downscale_time)
