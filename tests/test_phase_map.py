import os
import pytest
import numpy as np
import imageio.v3 as iio
from utils.config_loader import Config
import phase_map.cleanup as cleanup
import phase_map.grains as grains


def test_phases_selected():
    config = {
        "labels": ["A", "B", "C"],
        "detect": [True, False, True]
    }
    selected = cleanup._phases_selected("labels", config)
    assert selected == ["A", "C"]


def test_color_distance():
    # 2x2 image with 3 channels
    img_lab = np.zeros((2, 2, 3), dtype=np.float64)
    img_lab[0, 0] = [50, 10, 20]
    img_lab[0, 1] = [100, 0, 0]
    img_lab[1, 0] = [0, 0, 0]
    img_lab[1, 1] = [50, -10, -20]

    ref_colors_lab = np.array([
        [50, 10, 20],
        [100, 0, 0]
    ], dtype=np.float64)

    # test mode="lab"
    dist_lab = cleanup.color_distance(img_lab, ref_colors_lab, mode="lab")
    assert dist_lab.shape == (2, 2, 2)
    assert dist_lab[0, 0, 0] == 0

    # test mode="ab"
    dist_ab = cleanup.color_distance(img_lab, ref_colors_lab, mode="ab")
    assert dist_ab.shape == (2, 2, 2)
    assert dist_ab[0, 0, 0] == 0

    # test mode=None (CIE DE2000)
    dist_ciede = cleanup.color_distance(img_lab, ref_colors_lab, mode=None)
    assert dist_ciede.shape == (2, 2, 2)
    assert dist_ciede[0, 0, 0] == pytest.approx(0, abs=1e-3)


def test_apply_threshold(tmp_path):
    config = Config.load_config("config/phases_config.yaml", "phases")

    # Create synthetic RGB image with colors matching config
    h, w = 40, 40
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Set top-left quadrant to first detected color, top-right to second detected color
    detected_colors = cleanup._phases_selected("colors", config)
    rgb_img[:20, :20] = detected_colors[0]
    rgb_img[:20, 20:] = detected_colors[1]
    rgb_img[20:, :] = detected_colors[2]

    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    phase_map_res = cleanup.apply_threshold(
        phase_map=rgb_img,
        config=config,
        lightness_clip=[0, 100],
        output_dir=out_dir,
        output_file_prefix="test_"
    )
    assert phase_map_res.shape == (h, w)
    assert os.path.exists(os.path.join(out_dir, "test_phase_map.png"))
    assert os.path.exists(os.path.join(out_dir, "test_phases_config_used.json"))


def test_segment_grains_and_filter(tmp_path):
    # Create synthetic phase map with two large grains and small noise
    phase_map = np.zeros((100, 100), dtype=np.uint8)
    phase_map[10:45, 10:45] = 1
    phase_map[55:90, 55:90] = 2
    # Add tiny leftover
    phase_map[10:12, 50:52] = 1

    out_dir = str(tmp_path / "out_grains")
    os.makedirs(out_dir, exist_ok=True)

    grains_map = grains.segment_grains(phase_map, output_dir_path=out_dir, output_file_prefix="test_")
    assert grains_map.shape == (100, 100)
    assert os.path.exists(os.path.join(out_dir, "test_grain_map.tiff"))

    filtered_map, limit, non_matrix_count = grains.grain_size_filter(
        grains_map,
        grain_size_px_limit=500,
        output_dir_path=out_dir,
        output_file_prefix="test_"
    )
    assert filtered_map.shape == (100, 100)
    assert limit == 500
    assert os.path.exists(os.path.join(out_dir, "test_grain_map_filtered.tiff"))
