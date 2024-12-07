import conjugate

from pathlib import Path


ROOT = Path(__file__).parents[1]
PYPROJECT_TOML = ROOT / "pyproject.toml"


# Extract the current version from the pyproject.toml file
lines = PYPROJECT_TOML.read_text().split("\n")
CURRENT_VERSION = lines[2].split("=")[1].strip().replace('"', "")


def test_version():
    assert conjugate.__version__ == CURRENT_VERSION
