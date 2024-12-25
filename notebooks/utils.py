import sys
from pathlib import Path
import json


__CONFIG_FILE = "../config.json"


def mount_src():
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def load_config(path=__CONFIG_FILE) -> dict:
    with open(path, "r") as f:
        config = json.load(f)
    return config

