import pytest
import numpy as np
from cracks.derivates.bw_morph import thin, spur, branches, endpoints, _neighbors_conv
from cracks.derivates import smooth_mask, skeleton, skeleton_endpoints, skeleton_branches
from cracks.branches import sort_branch_pixels, path_direction, _find_path_start


def test_bw_morph_thin():
    # Square with center
    square = np.zeros((7, 7), dtype=np.uint8)
    square[1:-1, 2:-2] = 1
    square[0, 1] = 1
    thinned = thin(square)
    assert thinned.dtype == bool
    assert np.sum(thinned) < np.sum(square)
    assert np.sum(thinned) > 0


def test_bw_morph_spur():
    t = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0]
    ], dtype=np.uint8)
    spurred = spur(t)
    assert spurred.dtype == bool
    assert spurred.shape == t.shape


def test_bw_morph_branches_and_endpoints():
    # T-junction: pixel at (2, 2) connected to (1, 2), (3, 2), (2, 3), (2, 1) -> branch node
    cross = np.zeros((5, 5), dtype=np.uint8)
    cross[2, :] = 1
    cross[:, 2] = 1

    br = branches(cross)
    ep = endpoints(cross)

    assert br[2, 2] == True
    assert ep[0, 2] == True
    assert ep[4, 2] == True
    assert ep[2, 0] == True
    assert ep[2, 4] == True
    assert not ep[2, 2]


def test_bw_morph_invalid_input():
    with pytest.raises(ValueError):
        thin(np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        thin(np.zeros((3, 3)), n_iter=0)
    with pytest.raises(ValueError):
        thin(np.array([[0, 2], [1, 0]]))


def test_derivates_smooth_and_skeleton():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 10:40] = 1

    smooth = smooth_mask(mask, smooth_radius_px=2)
    assert smooth.shape == mask.shape

    smooth_0 = smooth_mask(mask, smooth_radius_px=0)
    np.testing.assert_array_equal(smooth_0, mask)

    skel_mask, distances = skeleton(smooth)
    assert skel_mask.shape == mask.shape
    assert np.sum(skel_mask) > 0
    assert distances.shape == mask.shape

    ep_mask, ep_coords = skeleton_endpoints(skel_mask)
    assert ep_mask.shape == mask.shape
    assert len(ep_coords) >= 2

    br_mask, nodes_mask, nodes_coords = skeleton_branches(skel_mask)
    assert br_mask.shape == mask.shape
    assert nodes_mask.shape == mask.shape


def test_branches_sort_and_direction():
    # Line of pixels
    line_mask = np.zeros((10, 10), dtype=np.uint8)
    line_mask[2, 2:8] = 1

    start_px = _find_path_start(line_mask)
    assert tuple(start_px) in [(2, 2), (2, 7)]

    ordered = sort_branch_pixels(line_mask)
    assert len(ordered) == 6
    assert ordered[0] in [(2, 2), (2, 7)]
    assert ordered[-1] in [(2, 2), (2, 7)]

    # Test path_direction
    grad = path_direction(ordered, line_mask)
    assert grad is not None
    assert grad.shape == (6, 2)


def test_branches_short_path():
    # Single pixel branch
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1
    # Only 1 pixel -> returns None
    grad = path_direction([(5, 5)], mask)
    assert grad is None
