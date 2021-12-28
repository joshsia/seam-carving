# Author: Joshua Sia
# Date: 2021-12-27
#
# Driver script for pipeline of seam carving
#
# Usage:
# make all

all : img/resized-image.jpg

# Download image
img/input-image.jpg :
	python src/get_image.py --url=https://www.vangoghgallery.com/img/starry_night_full.jpg

# Resize image
img/resized-image.jpg : img/input-image.jpg
	python src/resize_image.py --in_image=img/input-image.jpg

clean :
	rm -rf img/*