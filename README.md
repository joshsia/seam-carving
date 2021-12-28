# Seam carving

Author: Joshua Sia

Personal project inspired from UBC Master of Data Science Course DSCI 521.

### About

Seam carving is a content-aware image resizing technique which adds or removes rows or columns of pixels (known as horizontal or vertical seams) from an image iteratively based on contents of the image itself until the desired image size is achieved. The seam to be removed at each iteration is the one which contains pixels of the lowest summed energy which is a proxy for how important a given pixel is. A seam is valid if adjacent pixels are at most one pixel away in the vertical, horizontal or diagonal directions.

[This video](https://www.youtube.com/watch?v=6NcIJXTlugc) provides a demonstration of seam carving.

### Usage

There are two suggested ways to run this analysis:

#### 1\. Using Docker

*note - the instructions in this section also depends on running this in
a unix shell (e.g., terminal or Git Bash)*

To replicate the analysis, install
[Docker](https://www.docker.com/get-started). To pull the [Docker image](https://hub.docker.com/repository/docker/joshsia/seam-carving) from Docker Hub, run the following command:

```
docker pull joshsia/seam-carving
```

Clone this GitHub repository and run the following command at the command line/terminal
from the root directory of this project (Mac M1 users should add the flag and value `--platform linux/amd64`, and Windows users should use `//` in the path):

```
docker run --rm -it -v /$(pwd):/home joshsia/seam-carving make -C /home all
```

To reset the project to a clean state with no intermediate files, run the following command at the command line/terminal from the root directory of this project (Mac M1 users should add the flag and value `--platform linux/amd64`, and Windows users should use `//` in the path):

```
docker run --rm -it -v /$(pwd):/home joshsia/seam-carving make -C /home clean
```

#### 2\. Without using Docker

To replicate the analysis, clone this GitHub repository, install the dependencies listed below, and run the following command at the command line/terminal from the root directory of this project:

 ```
 make all
 ```

To reset the project to a clean state, with no intermediate or results files, run the following command at the command line/terminal from the root directory of this project:

 ```
 make clean
 ```

### Dependencies

- Python version 3.9.5 and Python packages:
    -   docopt=0.6.2
    -   requests=2.25.1
    -   numpy=1.21.2
    -   matplotlib=3.4.3
    -   scipy=1.7.1

### References

Avidan, S. and Shamir, A., 2007. Seam carving for content-aware image resizing. In ACM SIGGRAPH 2007 papers (pp. 10-es).
