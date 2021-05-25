FROM ubuntu:latest
MAINTAINER Tucker Siegel (tgsiegel@umd.edu)

RUN apt-get update
RUN apt-get install -y gcc-8 g++-8
RUN apt-get install -y cmake

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 100

RUN update-alternatives --set gcc /usr/bin/gcc-8
RUN update-alternatives --set g++ /usr/bin/g++-8

RUN apt-get install -y libboost-all-dev libblas-dev liblapack-dev