"""
Every compute and figure script must parse (syntax + no obvious errors) without
executing. We use py_compile rather than import because the scripts run their
plotting/IO at module scope.
"""
import py_compile
from pathlib import Path
import pytest
from conftest import CODE

SCRIPTS = sorted((CODE / "compute").glob("*.py")) + sorted((CODE / "figures").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(script):
    py_compile.compile(str(script), doraise=True)


def test_expected_scripts_present():
    names = {p.name for p in (CODE / "figures").glob("make_*.py")}
    # the scripts that own the headline figures must exist
    for must in ["make_skill_horizon.py", "make_density_scatter.py",
                 "make_probabilistic_skill.py", "make_week6_case_studies.py",
                 "make_domain_and_bias.py", "make_regional_tables.py"]:
        assert must in names, f"missing figure script: {must}"
    n_compute = len(list((CODE / "compute").glob("*.py")))
    assert n_compute >= 7, f"expected >=7 compute steps, found {n_compute}"
