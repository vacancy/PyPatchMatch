#! /bin/bash
#
# travis.sh
# Copyright (C) 2020 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#

cvpackages=$(pkg-config --cflags --libs opencv)
cflags=-Icsrc/

mkdir -p bin
g++ csrc/masked_image.cpp csrc/nnf.cpp csrc/inpaint.cpp csrc/main_test.cpp -g ${cflags} ${cvpackages} -std=c++14 -o bin/test
