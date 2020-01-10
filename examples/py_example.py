#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/09/2020
#
# Distributed under terms of the MIT license.

from PIL import Image

import sys
sys.path.insert(0, '../')
import patch_match


if __name__ == '__main__':
    source = Image.open('./images/forest_pruned.bmp')
    result = patch_match.inpaint(source, patch_size=3)
    Image.fromarray(result).show()

