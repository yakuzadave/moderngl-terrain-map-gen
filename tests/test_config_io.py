import json

from src import TerrainConfig, load_config, save_config


def test_save_and_load_yaml_config(tmp_path):
    config = TerrainConfig(
        resolution=1024,
        seed=123,
        terrain_preset="mountains",
        export_obj=True,
    )

    path = tmp_path / "terrain.yaml"
    save_config(config, path)
    loaded = load_config(path)

    assert loaded.resolution == 1024
    assert loaded.seed == 123
    assert loaded.terrain_preset == "mountains"
    assert loaded.export_obj is True


def test_save_and_load_json_config(tmp_path):
    config = TerrainConfig(
        resolution=2048,
        seed=999,
        generator_type="hydraulic",
        seamless=True,
    )

    path = tmp_path / "terrain.json"
    save_config(config, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["resolution"] == 2048
    assert saved["generator_type"] == "hydraulic"

    loaded = load_config(path)
    assert loaded.resolution == 2048
    assert loaded.seed == 999
    assert loaded.generator_type == "hydraulic"
    assert loaded.seamless is True
