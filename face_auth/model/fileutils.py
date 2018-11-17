import json
import os
from pathlib import Path


def create_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def create_parent_dir(path: str) -> None:
    create_dir(os.path.dirname(path))


def save_json(data, path: str) -> None:
    with open(path, mode='w') as out_file:
        json.dump(data, out_file)


def load_json(path: str):
    with open(path, mode='r') as in_file:
        return json.load(in_file)
