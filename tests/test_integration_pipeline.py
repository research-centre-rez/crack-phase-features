import os
import subprocess
import sys
import numpy as np
import imageio.v3 as iio
import pytest
from utils.config_loader import Config


@pytest.fixture
def synthetic_sample_data(tmp_path):
    """
    Creates lightweight synthetic sample data inspired by the real SEM BSE and EDS images in workspace.
    Size is 120x120 pixels so pipelines run fast during testing.
    """
    size = 120
    # 1. SEM BSE image (grayscale with dark crack line and texture)
    # Background around 140-160
    rng = np.random.RandomState(42)
    bse_img = rng.randint(140, 160, (size, size), dtype=np.uint8)
    # Add dark crack line in the middle vertically
    bse_img[20:100, 58:62] = 10  # Very dark seed (< 50)
    bse_img[15:105, 56:64] = np.clip(bse_img[15:105, 56:64], 0, 70)  # Seed potential

    bse_path = os.path.join(tmp_path, "sample_bse.tif")
    iio.imwrite(bse_path, bse_img)

    # 2. Phase EDS image (RGB with colored grains)
    phases_cfg = Config.load_config("config/phases_config.yaml", "phases")
    colors = [c for c, d in zip(phases_cfg["colors"], phases_cfg["detect"]) if d]
    
    eds_img = np.zeros((size, size, 3), dtype=np.uint8)
    # Left grain: color 0
    eds_img[:, :55] = colors[0]
    # Right grain: color 1
    eds_img[:, 65:] = colors[1] if len(colors) > 1 else colors[0]
    # Center matrix / mixed: color 2
    eds_img[:, 55:65] = colors[2] if len(colors) > 2 else colors[0]

    eds_path = os.path.join(tmp_path, "sample_eds.png")
    iio.imwrite(eds_path, eds_img)

    return {
        "bse_path": bse_path,
        "eds_path": eds_path,
        "output_dir": str(tmp_path / "output")
    }


def test_full_pipeline_cli_integration(synthetic_sample_data):
    """
    Tests the complete end-to-end workflow as described in README.md:
    1. python -m cracks.detection
    2. python -m phase_map
    3. python full_crack_analysis.py -f crack_features
    4. python full_crack_analysis.py -f phase_features
    """
    bse_path = synthetic_sample_data["bse_path"]
    eds_path = synthetic_sample_data["eds_path"]
    out_dir = synthetic_sample_data["output_dir"]
    sample_name = "test_sample"
    python_bin = sys.executable

    # 1. Crack detection
    cmd_detection = [
        python_bin, "-m", "cracks.detection",
        "-o", out_dir,
        "--sample-name", sample_name,
        "-c", "config/cracks_config.yaml",
        bse_path
    ]
    res = subprocess.run(cmd_detection, capture_output=True, text=True)
    assert res.returncode == 0, f"Detection failed: {res.stderr}"

    sample_out_dir = os.path.join(out_dir, sample_name)
    crack_mask_path = os.path.join(sample_out_dir, f"{sample_name}-crack-mask.png")
    assert os.path.exists(crack_mask_path)
    assert os.path.exists(os.path.join(sample_out_dir, f"{sample_name}-[user]colored.png"))

    # 2. Phase map
    cmd_phase_map = [
        python_bin, "-m", "phase_map",
        "-o", out_dir,
        "--sample-name", sample_name,
        "-c", "config/phases_config.yaml",
        eds_path,
        crack_mask_path
    ]
    res = subprocess.run(cmd_phase_map, capture_output=True, text=True)
    assert res.returncode == 0, f"Phase map failed: {res.stderr}"

    phase_map_filtered_path = os.path.join(sample_out_dir, f"{sample_name}-phase_map_filtered.png")
    assert os.path.exists(phase_map_filtered_path)
    assert os.path.exists(os.path.join(sample_out_dir, f"{sample_name}-grain_map_filtered.tiff"))

    # 3. Full crack analysis: crack_features
    cmd_analysis_crack = [
        python_bin, "full_crack_analysis.py",
        "-f", "crack_features",
        "--sample-name", sample_name,
        "-o", out_dir,
        "-c", "config/phases_config.yaml",
        crack_mask_path
    ]
    res = subprocess.run(cmd_analysis_crack, capture_output=True, text=True)
    assert res.returncode == 0, f"Analysis crack features failed: {res.stderr}"
    assert os.path.exists(os.path.join(sample_out_dir, f"{sample_name}-cracks_features.csv"))

    # 4. Full crack analysis: phase_features
    cmd_analysis_phase = [
        python_bin, "full_crack_analysis.py",
        "-f", "phase_features",
        "--sample-name", sample_name,
        "-o", out_dir,
        "-c", "config/phases_config.yaml",
        "--phase-map-path", phase_map_filtered_path,
        crack_mask_path
    ]
    res = subprocess.run(cmd_analysis_phase, capture_output=True, text=True)
    assert res.returncode == 0, f"Analysis phase features failed: {res.stderr}"
    assert os.path.exists(os.path.join(sample_out_dir, f"{sample_name}-phase-through.png"))
    assert os.path.exists(os.path.join(sample_out_dir, f"{sample_name}-phase-edge.png"))
    assert os.path.exists(os.path.join(sample_out_dir, f"{sample_name}-phases_cracks.csv"))
    assert os.path.exists(os.path.join(sample_out_dir, f"{sample_name}-skeleton-neighbors.csv"))
