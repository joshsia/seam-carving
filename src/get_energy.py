# Author: Joshua Sia
# Date: 2021-12-27

'''This script computes the energy function for an image. The 
script takes the path of the input image, and an optional filename to save
the output as.

Usage:
get_energy.py --in_file=<in_file> [--out_file=<out_file>]

Options:
--in_file=<in_file>     URL to download text from
--out_file=<out_file>   Filename to save corpus as [default: input-energy.npy]
'''

# python src/get_energy.py --in_file=img/input-image.jpg

import sys
from docopt import docopt
import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt

opt = docopt(__doc__)

def main(opt):

    img = plt.imread(opt["--in_file"])

    dy = np.array([-1, 0, 1])[:, None, None]
    dx = np.array([-1, 0, 1])[None, :, None]

    energy_img = np.sum(convolve(img, dx)**2 + convolve(img, dy)**2, axis=2)

    out_file = "img/" + opt["--out_file"]

    np.save(out_file, energy_img)

if __name__ == "__main__":
  main(opt)