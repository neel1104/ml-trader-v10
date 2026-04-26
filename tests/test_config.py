import json
import os

def test_config_validity():
    config_path = "user_data/config.json"
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)
    assert config["dry_run"] is True
    assert config["freqai"]["enabled"] is True
    assert "BTC/USDT:USDT" in config["exchange"]["pair_whitelist"]
