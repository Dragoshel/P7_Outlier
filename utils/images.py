import torchvision
import matplotlib.pyplot as plt
import numpy as np

from utils.normalizer import get_norm_factor

def show_images(images: list, one_channel: bool = False) -> None:
    """ Display images in the dataset in a 4x4 grid, before displaying the images
    they need to be unnormalised again

    Args:
        images (list): The images to display in the grid
        one_channel (bool, optional): Whether or not the images are greyscale. Defaults to False.
    """    
    image_grid = torchvision.utils.make_grid(images)
    if one_channel:
        image_grid = image_grid.mean(dim=0)
    image_grid = image_grid / 2 + get_norm_factor()
    np_grid = image_grid.numpy()
    if one_channel:
        plt.imshow(np_grid, cmap="Greys")
    else:
        plt.imshow(np.transpose(np_grid, (1, 2, 0)))
        