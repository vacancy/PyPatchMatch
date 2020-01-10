#! /bin/bash
#
# cpp_example_run.sh
# Copyright (C) 2020 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#

set -x

CFLAGS="-std=c++14 -O2 $(pkg-config --libs --cflags opencv)"
g++ $CFLAGS cpp_example.cpp -L../ -lpatchmatch -I../csrc/ -o cpp_example.exe
LD_LIBRARY_PATH=../:$LD_LIBRARY_PATH ./cpp_example.exe

