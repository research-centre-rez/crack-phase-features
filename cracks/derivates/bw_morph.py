"""
Functions that implement some of the same functionality found in Matlab's bwmorph.

`thin` - was taken and adapted from https://gist.github.com/joefutrelle/562f25bbcf20691217b8
`spur` - Not perfect but pretty close to what matlab does via LUTs
`endpoints` - lines up perfectly with matlab's output (in my limited testing)
`branches` - this results in more clustered pixels than matlab's version but it pretty close
"""
from typing import Optional, Sequence
import numpy as np
import scipy.ndimage as ndi

# Binary lookup table kernel mask:
# Each 3x3 neighboring cell is weighted with a unique power of 2,
# allowing the 8-neighborhood binary pattern to be encoded into an 8-bit integer [0..255].
LUT_DEL_MASK = np.array([
    [8, 4, 2],
    [16, 0, 1],
    [32, 64, 128]
], dtype=np.uint8)


def _bwmorph_luts(image: np.ndarray, luts: Sequence[np.ndarray], n_iter: Optional[int] = None, padding: int = 0) -> np.ndarray:
    """
    Perform LUT-based binary morphological thinning or spur removal.

    :param image: 2D binary numpy array (elements must be 0 or 1).
    :param luts: List of 256-element boolean lookup tables.
    :param n_iter: Maximum number of iterations (None for indefinite until convergence).
    :param padding: Padding constant value along the image border.
    :return: Processed binary mask as a boolean ndarray.
    """
    # check parameters
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter

    # check that we have a 2d binary image, and convert it to uint8
    im = np.array(image).astype(np.uint8)

    if im.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.isin(image.flat, (0, 1))):
        raise ValueError('Image contains values other than 0 and 1')

    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(im)  # count points before

        # for each subiteration
        for lut in luts:
            # correlate image with neighborhood mask to compute 8-bit neighborhood code for each pixel
            neighborhood_codes = ndi.correlate(im, LUT_DEL_MASK, mode='constant', cval=padding)
            # lookup deletion decision (True/False) from this subiteration's LUT
            delete_mask = np.take(lut, neighborhood_codes)
            # perform deletion by setting matching pixels to 0
            im[delete_mask] = 0

        after = np.sum(im)  # count points after

        if before == after:
            # iteration had no effect: finish
            break

        # count down to iteration limit (or endlessly negative)
        n -= 1

    return im.astype(bool)


# lookup tables for thin
G123_LUT = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
    0, 0, 0
], dtype=bool)

G123P_LUT = np.array([
    0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
    0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0
], dtype=bool)

THIN_LUTS = [G123_LUT, G123P_LUT]


def thin(image: np.ndarray, n_iter: Optional[int] = None) -> np.ndarray:
    """
    Perform morphological thinning of a binary image.

    :param image: Binary 2D ndarray to be thinned.
    :param n_iter: Optional int upper bound on the number of thinning passes.
    :return: Thinned image as boolean ndarray.
    """
    return _bwmorph_luts(image, THIN_LUTS, n_iter=n_iter)


SPUR_LUT = np.array([
    1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=bool)


def spur(image: np.ndarray, n_iter: Optional[int] = None) -> np.ndarray:
    """
    Remove small spur branches from a binary image skeleton.

    :param image: Binary 2D ndarray.
    :param n_iter: Optional int upper bound on number of spur iterations.
    :return: De-spurred boolean ndarray image.
    """
    return _bwmorph_luts(image, [SPUR_LUT], n_iter=n_iter, padding=1)


def _neighbors_conv(image: np.ndarray) -> np.ndarray:
    """
    Count the number of 8-connected neighbors for each non-zero pixel in an image.

    :param image: 2D binary numpy array.
    :return: ndarray of neighbor counts for each pixel.
    """
    # 3x3 kernel counting all 8 surrounding neighbors (excluding the center pixel)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighborhood_count = ndi.convolve(image.astype(int), k, mode='constant', cval=0)
    # Zero out background pixels so counts only apply to foreground pixels
    neighborhood_count[np.logical_not(image)] = 0
    return neighborhood_count


def branches(image: np.ndarray) -> np.ndarray:
    """
    Return binary mask of branch nodes (pixels with > 2 neighbors in skeleton).

    :param image: Binary 2D ndarray.
    :return: Boolean ndarray mask of branch nodes.
    """
    return _neighbors_conv(image) > 2


def endpoints(image: np.ndarray) -> np.ndarray:
    """
    Return binary mask of endpoints (pixels with exactly 1 neighbor in skeleton).

    :param image: Binary 2D ndarray.
    :return: Boolean ndarray mask of skeleton endpoints.
    """
    return _neighbors_conv(image) == 1