import os
import pytest
import numpy as np
import imageio.v3 as iio
from utils.config_loader import Config
import cracks.detection as detection
import cracks.features as features


def test_detection_load_input(tmp_path):
    # Test grayscale uint8
    gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    p_gray = os.path.join(tmp_path, "gray.png")
    iio.imwrite(p_gray, gray)
    loaded = detection._load_input(p_gray)
    np.testing.assert_array_equal(loaded, gray)

    # Test 16-bit
    u16 = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
    p_u16 = os.path.join(tmp_path, "u16.png")
    iio.imwrite(p_u16, u16)
    loaded_16 = detection._load_input(p_u16)
    assert loaded_16.dtype == np.uint8

    # Test RGB
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    p_rgb = os.path.join(tmp_path, "rgb.png")
    iio.imwrite(p_rgb, rgb)
    loaded_rgb = detection._load_input(p_rgb)
    assert loaded_rgb.shape == (10, 10)

    # Test RGBA
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[:, :, 0] = 200
    rgba[:, :, 3] = 255
    p_rgba = os.path.join(tmp_path, "rgba.png")
    iio.imwrite(p_rgba, rgba)
    loaded_rgba = detection._load_input(p_rgba)
    assert loaded_rgba.shape == (10, 10)


def test_normalize_histogram_left_and_right():
    img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    norm = detection.normalize_histogram_left_and_right(img, 128)
    assert norm.shape == (50, 50)


def test_crack_mask_by_thresholds(tmp_path):
    config = Config.load_config("config/cracks_config.yaml", "cracks")
    detection.CRACKS_CONFIG = config

    # Create synthetic image with dark crack lines
    img = 150 * np.ones((50, 50), dtype=np.uint8)
    img[20:30, 25] = 10  # dark crack seed

    img_path = os.path.join(tmp_path, "synthetic_crack_input.png")
    iio.imwrite(img_path, img)

    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    mask = detection.crack_mask_by_thresholds(img_path, out_dir, output_file_prefix="test_")
    assert mask.shape == (50, 50)
    assert os.path.exists(os.path.join(out_dir, "test_crack-mask.png"))


def test_compute_cracks_empty_mask():
    empty_mask = np.zeros((50, 50), dtype=np.uint8)
    res = features.compute_cracks(empty_mask)
    assert res == []


def test_compute_cracks_synthetic_crack():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[20:40, 28:32] = 1  # a rectangular crack

    res = features.compute_cracks(mask)
    assert len(res) == 1
    crack_info = res[0]
    assert crack_info["label"] == 1
    assert crack_info["crackSize_px"] == 20 * 4
    assert crack_info["skeleton_px"] > 0
    assert crack_info["maxWidth_px"] > 0
    assert crack_info["avgWidth_px"] > 0
    assert crack_info["boundaryLength_px"] > 0
    assert crack_info["farthestPoints_px"] > 0


def test_phase_analysis():
    # Synthetic crack mask and phase map
    crack_mask = np.zeros((60, 60), dtype=np.uint8)
    crack_mask[20:40, 30] = 1

    phase_map = np.ones((60, 60), dtype=np.int32)
    # Left side phase 1, right side phase 2
    phase_map[:, :30] = 1
    phase_map[:, 31:] = 2

    through, edge, neighbors_map, branches_map, gradient_map = features.phase_analysis(crack_mask, phase_map)

    assert through.shape == crack_mask.shape
    assert edge.shape == crack_mask.shape + (2,)
    assert neighbors_map.shape == crack_mask.shape + (2, 2)
    assert branches_map.shape == crack_mask.shape
    assert gradient_map.shape == crack_mask.shape + (2,)
