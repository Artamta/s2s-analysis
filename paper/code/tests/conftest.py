"""
Shared paths and fixtures for the test suite.

Paths are derived from this file's location so the tests are portable:
    paper/code/tests/conftest.py  ->  parents[3] == repository root
The figure scripts read precomputed skill tables from analysis-code/analysis/
and write figures to paper/figs/; the tests check exactly those locations.
"""
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "paper" / "code"
DATA = REPO / "analysis-code" / "analysis"            # precomputed skill tables
FIGS = REPO / "paper" / "figs"                        # manuscript figures
TEX = REPO / "paper" / "jfm2026_india_s2s_benchmark.tex"

# make `from utils...` resolve to the package's own utils/
sys.path.insert(0, str(CODE))

MODELS = ["SPIRE", "FuXi", "ECMWF", "NCEP"]
N_INITS = 13
WEEKS = [f"Week {i}" for i in range(1, 7)]
REGIONS = {"All India", "northwest_india", "central_india",
           "south_peninsula", "east_northeast_india"}


def require(path):
    """Skip a test (rather than fail) if a precomputed input is absent."""
    if not Path(path).exists():
        pytest.skip(f"required input not found: {path}")
    return Path(path)


@pytest.fixture(scope="session")
def data_dir():
    if not DATA.exists():
        pytest.skip(f"data directory not found: {DATA}")
    return DATA
