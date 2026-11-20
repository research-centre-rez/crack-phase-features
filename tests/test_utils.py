import os
import pytest
import numpy as np
import imageio.v3 as iio
from utils.config_loader import Config
import utils.image_logger as image_logger
import utils.output_dir_generator as output_generator


def test_config_loader_cracks_yaml():
    config = Config.load_config("config/cracks_config.yaml", "cracks")
    assert "median_level" in config
    assert "threshold_levels" in config
    assert Config.get_config("cracks") == config


def test_config_loader_phases_yaml():
    config = Config.load_config("config/phases_config.yaml", "phases")
    assert "labels" in config
    assert "colors" in config
    assert "detect" in config
    assert Config.get_config("phases") == config


def test_config_loader_invalid_type(tmp_path):
    with pytest.raises(Exception):
        Config.load_config("config/cracks_config.yaml", "invalid_type")

    with pytest.raises(Exception):
        Config.get_config("invalid_type")


def test_config_loader_json_format(tmp_path):
    json_file = tmp_path / "test_config.json"
    json_file.write_text('{"median_level": 128, "threshold_levels": []}')
    cfg = Config.load_config(str(json_file), "cracks")
    assert cfg["median_level"] == 128


def test_output_dir_generator(tmp_path):
    out_dir = tmp_path / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    sub_dir = out_dir / "sub"
    # parent directory exists
    res_dir, prefix = output_generator.prepare_output_path(str(sub_dir), sample_name="sample1")
    assert os.path.exists(res_dir)
    assert prefix == "sample1-"
    assert res_dir.endswith(os.path.join("sub", "sample1"))

    # without sample_name
    res_dir2, prefix2 = output_generator.prepare_output_path(str(sub_dir))
    assert prefix2 == ""
    assert res_dir2 == str(sub_dir)


def test_output_dir_generator_nonexistent_parent():
    with pytest.raises(Exception):
        output_generator.prepare_output_path("/non/existent/path/for/tests/output")


def test_image_logger(tmp_path):
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:5, 2:5] = 255

    dump_path = os.path.join(tmp_path, "test_img.png")
    image_logger.dump_image(dump_path, img)
    assert os.path.exists(dump_path)
    loaded = iio.imread(dump_path)
    assert loaded.shape == (10, 10)

    # Test normalization when dtype != uint8
    img_float = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
    float_path = os.path.join(tmp_path, "test_float.png")
    image_logger.dump_image(float_path, img_float)
    assert os.path.exists(float_path)

    # Test debug, info, warning, error functions
    image_logger.info(img, str(tmp_path), "info.png")
    assert os.path.exists(os.path.join(tmp_path, "info.png"))
    image_logger.warning(img, str(tmp_path), "warn.png")
    assert os.path.exists(os.path.join(tmp_path, "warn.png"))
    image_logger.error(img, str(tmp_path), "err.png")
    assert os.path.exists(os.path.join(tmp_path, "err.png"))
