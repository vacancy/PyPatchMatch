#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : patch_match.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/09/2020
#
# Distributed under terms of the MIT license.

import ctypes
import os.path as osp
from typing import Optional, Union

import numpy as np
from PIL import Image

__all__ = ['inpaint']


class CShapeT(ctypes.Structure):
    _fields_ = [
        ('width', ctypes.c_int),
        ('height', ctypes.c_int),
        ('channels', ctypes.c_int),
    ]


class CMatT(ctypes.Structure):
    _fields_ = [
        ('data_ptr', ctypes.c_void_p),
        ('shape', CShapeT),
        ('dtype', ctypes.c_int)
    ]


PMLIB = ctypes.CDLL(osp.join(osp.dirname(__file__), 'libpatchmatch.so'))

PMLIB.PM_inpaint.argtypes = [CMatT, CMatT, ctypes.c_int]
PMLIB.PM_inpaint.restype = CMatT
PMLIB.PM_free_pymat.argtypes = [CMatT]


def inpaint(
    image: Union[np.ndarray, Image.Image],
    mask: Optional[Union[np.ndarray, Image.Image]] = None,
    patch_size: int = 15
) -> np.ndarray:
    """
    PatchMatch based inpainting proposed in:

        PatchMatch : A Randomized Correspondence Algorithm for Structural Image Editing
        C.Barnes, E.Shechtman, A.Finkelstein and Dan B.Goldman
        SIGGRAPH 2009

    Args:
        image (Union[np.ndarray, Image.Image]): the input image, should be 3-channel RGB/BGR.
        mask (Union[np.array, Image.Image], optional): the mask of the hole(s) to be filled, should be 1-channel.
        If not provided (None), the algorithm will treat all purely white pixels as the holes (255, 255, 255).
        patch_size (int): the patch size for the inpainting algorithm.

    Return:
        result (np.ndarray): the repaired image, of the same size as the input image.
    """

    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == 'uint8'
    if mask is None:
        mask = (image == (255, 255, 255)).all(axis=2, keepdims=True).astype('uint8')
    assert mask.ndim == 3 and mask.shape[2] == 1 and mask.dtype == 'uint8'

    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)

    ret_pymat = PMLIB.PM_inpaint(np_to_pymat(image), np_to_pymat(mask), ctypes.c_int(patch_size))
    ret_npmat = pymat_to_np(ret_pymat)
    PMLIB.PM_free_pymat(ret_pymat)

    return ret_npmat


dtype_pymat_to_ctypes = [
    ctypes.c_uint8,
    ctypes.c_int8,
    ctypes.c_uint16,
    ctypes.c_int16,
    ctypes.c_int32,
    ctypes.c_float,
    ctypes.c_double,
]


dtype_np_to_pymat = {
    'uint8': 0,
    'int8': 1,
    'uint16': 2,
    'int16': 3,
    'int32': 4,
    'float32': 5,
    'float64': 6,
}



def np_to_pymat(npmat):
    assert npmat.ndim == 3
    return CMatT(
        ctypes.cast(npmat.ctypes.data, ctypes.c_void_p),
        CShapeT(npmat.shape[1], npmat.shape[0], npmat.shape[2]),
        dtype_np_to_pymat[str(npmat.dtype)]
    )


def pymat_to_np(pymat):
    npmat = np.ctypeslib.as_array(
        ctypes.cast(pymat.data_ptr, ctypes.POINTER(dtype_pymat_to_ctypes[pymat.dtype])),
        (pymat.shape.height, pymat.shape.width, pymat.shape.channels)
    )
    ret = np.empty(npmat.shape, npmat.dtype)
    ret[:] = npmat
    return ret

