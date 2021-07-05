import os
import subprocess
from network.arch import _register_arch

_ = _register_arch

this_directory = os.path.abspath(os.path.dirname(__file__))
PROJECT_PATH = os.path.dirname(this_directory)
DATA_PATH = os.path.join(PROJECT_PATH, ".data")
CONFIG_PATH = os.path.join(PROJECT_PATH, "config")
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

__git_hash__ = (
    subprocess.check_output([f"cd {PROJECT_PATH}; git rev-parse HEAD"], shell=True)
    .strip()
    .decode()
)


def adding_hash(config, **kwargs):
    return {**config, **kwargs}
