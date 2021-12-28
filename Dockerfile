# Author: Joshua Sia
# Date: 2021-12-27

FROM continuumio/anaconda3:2021.11

RUN apt-get update

RUN apt-get install make

RUN conda install docopt=0.6.2 -y