# Seam carving

Author: Joshua Sia

Personal project inspired from UBC Master of Data Science Course DSCI 521.

### About

Seam carving is a content-aware image resizing technique which adds or removes rows or columns of pixels (known as horizontal or vertical seams) from an image iteratively based on contents of the image itself until the desired image size is achieved. The seam to be removed at each iteration is the one which contains pixels of the lowest summed energy which is a proxy for how important a given pixel is. A seam is valid if adjacent pixels are at most one pixel away in the vertical, horizontal or diagonal directions.

[This video](https://www.youtube.com/watch?v=6NcIJXTlugc) provides a demonstration of seam carving.

### Usage

### Dependencies

### References

Avidan, S. and Shamir, A., 2007. Seam carving for content-aware image resizing. In ACM SIGGRAPH 2007 papers (pp. 10-es).
