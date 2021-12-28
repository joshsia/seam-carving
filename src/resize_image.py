# Author: Joshua Sia
# Date: 2021-12-27

'''This script resizes an image based on the energy function of
the image. The script takes the path of the input energy function
of the image, and an optional filename to save
the output as.

Usage:
get_energy.py --in_image=<in_image> [--height=<height>] [--width=<width>] [--out_file=<out_file>]

Options:
--in_image=<in_image>       Path to input image
--height=<height>           Ratio of new height to original [default: 0.5]
--width=<width>             Ratio of new height to original [default: 0.5]
--out_file=<out_file>       Filename of resized image [default: resized-image.jpg]
'''

# python src/resize_image.py --in_image=img/input-image.jpg

import sys
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

opt = docopt(__doc__)

def main(opt):

    img = plt.imread(opt["--in_image"])
    resized_img = seam_carve(img, float(opt["--height"]), float(opt["--width"]))

    out_file = "img/" + opt["--out_file"]

    plt.imsave(out_file, resized_img)


def find_vertical_seam(energy):
    """
    The vertical seam of lowest total energy of the image.

    Parameters
    ----------
    energy : numpy.ndarray
        the 2d energy image

    Returns
    -------
    list 
        the seam of column indices

    Example
    -------
    e = np.array([[0.6625, 0.3939], [1.0069, 0.7383]])
    find_vertical_seam(e)
    [1, 1]
    """

    nrows, ncols = energy.shape

    # Cumulative Minimum Energy
    CME = np.zeros((nrows, ncols+2))
    CME[:,0] = np.inf
    CME[:,-1] = np.inf
    CME[:,1:-1] = energy

    for i in range(1, nrows):
        temp = np.full((3, ncols+2), np.inf)
        temp[0, 2:] = CME[i-1, 1:-1]
        temp[1, 1:-1] = CME[i-1, 1:-1]
        temp[2, :-2] = CME[i-1, 1:-1]

        CME[i] += np.min(temp, axis=0)

    # Create the seam array
    seam = np.zeros(nrows, dtype=int)
    seam[-1] = np.argmin(CME[-1,:])

    # Track the path backwards
    for i in range(nrows-2,-1,-1):
        # min_index is 0, 1, or 2. Subtract 1 to give the offset from
        # seam(i+1), namely -1, 0, or 1. Then add this to the old value.
        delta = np.argmin(CME[i, seam[i+1]-1:seam[i+1]+2]) - 1
        seam[i] = seam[i+1] + delta

    return seam - 1 
    # -1 because the indices are all off by 1, due to padding of CME


def find_horizontal_seam(energy):
    """
    Find the minimum-energy horizontal seam in an image. 

    Parameters
    ----------
    energy : numpy.ndarray
        a 2d numpy array containing the energy values. Size NxM

    Returns
    -------
    numpy.ndarray
        a seam represented as a 1d array of length M, with all values between 0 and N-1
    """

    return find_vertical_seam(energy.T)


def remove_vertical_seam(image, seam):
    """
    Remove a vertical seam from an image.

    Parameters
    ----------
    image : numpy.ndarray
        a 3d numpy array containing the pixel values
    seam : numpy.ndarray
        a 1d array (or list) containing the column index of each pixel in the seam
        length N, all values between 0 and M-1

    Returns
    -------
    numpy.ndarray
        a new image that is smaller by 1 column. Size N by M-1.
    """

    height = image.shape[0]
    linear_inds = np.array(seam)+np.arange(image.shape[0])*image.shape[1]
    new_image = np.zeros(
        (height, image.shape[1]-1, image.shape[-1]), dtype=image.dtype)
    for c in range(image.shape[-1]):
        temp = np.delete(image[:, :, c], linear_inds.astype(int))
        temp = np.reshape(temp, (height, image.shape[1]-1))
        new_image[:, :, c] = temp
    return new_image


def remove_horizontal_seam(image, seam):
    """
    Remove a horizontal seam from an image.

    Parameters
    ----------
    image : numpy.ndarray 
        a 2d numpy array containing the pixel values. Size NxM
    seam : numpy.ndarray
        a 1d array containing the column index of each pixel in the seam
        length N, all values between 0 and M-1.

    Returns
    -------
    numpy.ndarray
        a new image that is smaller by 1 row. Size N-1 by M.
    """

    return np.transpose(remove_vertical_seam(np.transpose(image, (1, 0, 2)), seam), (1, 0, 2))


def seam_carve(image, desired_height, desired_width):
    """
    Resize an NxM image to a desired height and width based on the
    energy function of the image. 
    Note: this function only makes images smaller. Enlarging an image is not implemented. 
    Note: this function removes all vertical seams before removing any horizontal
    seams, which may not be optimal.

    Parameters
    ----------
    image : numpy.ndarray
        a 3d numpy array of size N x M x 3
    desired_width : float
        the desired ratio of new to old width
    desired_height : float 
        the desired ratio of new to old height
        
    Returns
    -------
    numpy array
        the resized image
    """

    original_width = image.shape[1]
    original_height = image.shape[0]

    # Removing vertical seams (i.e. decreasing width)
    while image.shape[1] > (desired_width * original_width):
        seam = find_vertical_seam(get_energy(image))
        assert len(seam) == image.shape[0], "the length of the seam must equal the height of the image"
        
        image = remove_vertical_seam(image, seam)
        print(f"\rWidth is now {round(100 * image.shape[1] / original_width, 1)}% of original", end="")

    # Removing horizontal seams (i.e. decreasing height)
    while image.shape[0] > (desired_height * original_height):
        seam = find_horizontal_seam(get_energy(image))
        assert len(seam) == image.shape[1], "the length of the seam must equal the width of the image"

        image = remove_horizontal_seam(image, seam)
        print(f"\rHeight is now {round(100 * image.shape[0] / original_height, 1)}% of original", end="")
    
    return image


def get_energy(image):
    """
    Computes the energy function of an image.

    Parameters
    ----------
    image : numpy.ndarray
        a 3d numpy array of size N x M x 3
        
    Returns
    -------
    numpy.ndarray
        A new image where the pixels values represent the energy
        of the corresponding pixel in the original image
    """

    dy = np.array([-1, 0, 1])[:, None, None]
    dx = np.array([-1, 0, 1])[None, :, None]

    return np.sum(convolve(image, dx)**2 + convolve(image, dy)**2, axis=2)


if __name__ == "__main__":
  main(opt)