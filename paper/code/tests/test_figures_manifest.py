"""
Consistency between the manuscript and the figures on disk: every figure the
.tex includes must exist in paper/figs and be a non-trivial file.
"""
import re
import pytest
from conftest import TEX, FIGS, require

require(TEX)
TEX_SRC = TEX.read_text(encoding="utf-8", errors="ignore")
INCLUDED = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", TEX_SRC)


def test_manuscript_includes_figures():
    assert len(INCLUDED) >= 20, f"only {len(INCLUDED)} \\includegraphics found"


@pytest.mark.parametrize("ref", INCLUDED, ids=lambda s: s.split("/")[-1])
def test_each_included_figure_exists(ref):
    name = ref.split("/")[-1]
    name = name if name.endswith(".pdf") else name + ".pdf"
    f = FIGS / name
    assert f.exists(), f"figure referenced by .tex missing: {f}"
    assert f.stat().st_size > 1024, f"figure suspiciously small: {f}"


def test_headline_figures_present():
    for stem in ["fig02_skill_horizon", "fig11_scatter_density",
                 "fig15_crpss", "fig16_spread_skill",
                 "fig20_casestudy_t2m", "fig04_regional_scorecard"]:
        assert (FIGS / f"{stem}.pdf").exists(), f"missing headline figure {stem}.pdf"


def test_no_removed_taylor_figure_referenced():
    """The Taylor diagram was removed from the manuscript; it must not be cited."""
    assert "fig14_taylor" not in TEX_SRC
    assert "fig:taylor" not in TEX_SRC
