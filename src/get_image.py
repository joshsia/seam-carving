# Author: Joshua Sia
# Date: 2021-12-27

'''This script downloads an image from a given URL. The script
takes a URL, and an optional filename to save as.

Usage:
get_image.py --url=<url> [--out_file=<out_file>]

Options:
--url=<url>             URL to download text from
--out_file=<out_file>   Filename to save corpus as [default: input-image.jpg]
'''

# python src/get_image.py --url=https://www.vangoghgallery.com/img/starry_night_full.jpg
# python src/get_image.py --url=http://ubc-mds.github.io/img/mdslogopad.png
# python src/get_image.py --url=https://www.jerrysartarama.com/blog/images/2015/05/blog-mona-lisa-1.jpg
# python src/get_image.py --url=https://www.adorama.com/alc/wp-content/uploads/2018/11/landscape-photography-tips-yosemite-valley-feature.jpg

import sys
from docopt import docopt
import requests

opt = docopt(__doc__)

def main(opt):

    img = requests.get(opt["--url"]).content

    out_file = "img/" + opt["--out_file"]

    with open(out_file, 'wb') as f:
        f.write(img)

if __name__ == "__main__":
  main(opt)