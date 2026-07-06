#!/usr/bin/env python3
r"""Assemble a clean arXiv submission tarball for the two-season India S2S paper.

arXiv compiles with its own TeX Live (pdflatex/latexmk), *not* Tectonic, and it
does not run BibTeX unless a .bbl is present. This paper sidesteps BibTeX
entirely by carrying a manual ``thebibliography`` block (bibliography_block.tex),
so a self-contained source tarball is all arXiv needs.

This script copies only the files the main .tex actually depends on
(the main source, the manual bibliography, every \input'd table, and every
\includegraphics figure), scrubs build intermediates, and produces:

    paper_v2/arxiv_submission/            staged, compilable source tree
    paper_v2/arxiv_submission.tar.gz      the upload tarball

It parses the dependency list out of the .tex rather than hard-coding it, so
adding a table/figure to the paper does not silently drop it from the bundle.

Run:
    python paper_v2/scripts/make_arxiv_bundle.py
Then upload arxiv_submission.tar.gz, or test locally with:
    cd arxiv_submission && latexmk -pdf s2s_india_benchmark.tex
"""
from __future__ import annotations

import os
import re
import shutil
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.abspath(os.path.join(HERE, ".."))
MAIN_TEX = "s2s_india_benchmark.tex"
STAGE = os.path.join(PAPER_DIR, "arxiv_submission")
TARBALL = os.path.join(PAPER_DIR, "arxiv_submission.tar.gz")

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def _resolve(ref: str, exts: list[str]) -> str:
    """Resolve a LaTeX reference (which may omit its extension) to a real file
    under PAPER_DIR, trying the given extensions in order."""
    cand = os.path.join(PAPER_DIR, ref)
    if os.path.exists(cand):
        return ref
    for ext in exts:
        if os.path.exists(cand + ext):
            return ref + ext
    raise FileNotFoundError(f"cannot resolve referenced file: {ref}")


def collect_dependencies() -> list[str]:
    text = open(os.path.join(PAPER_DIR, MAIN_TEX)).read()
    deps = {MAIN_TEX}
    for ref in INPUT_RE.findall(text):
        deps.add(_resolve(ref, [".tex"]))
    for ref in GRAPHIC_RE.findall(text):
        deps.add(_resolve(ref, [".pdf", ".png", ".jpg", ".jpeg"]))
    return sorted(deps)


def main() -> None:
    deps = collect_dependencies()

    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    for rel in deps:
        src = os.path.join(PAPER_DIR, rel)
        dst = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    # A short arXiv note (ignored by the compiler, helpful to a human reviewer).
    with open(os.path.join(STAGE, "00README.txt"), "w") as fh:
        fh.write(
            "arXiv source bundle for:\n"
            "  When Does Machine Learning Add Subseasonal Forecast Skill over India?\n"
            "  An Early Benchmark across Winter and Monsoon Regimes\n\n"
            f"Main file: {MAIN_TEX}\n"
            "Compile with: latexmk -pdf s2s_india_benchmark.tex\n"
            "Bibliography is a manual thebibliography block "
            "(bibliography_block.tex); no BibTeX run is required.\n"
        )

    with tarfile.open(TARBALL, "w:gz") as tar:
        for root, _dirs, files in os.walk(STAGE):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, STAGE)
                tar.add(full, arcname=arc)

    size_mb = os.path.getsize(TARBALL) / 1e6
    print(f"staged {len(deps)} source files + 00README under {STAGE}")
    for rel in deps:
        print(f"  {rel}")
    print(f"\nwrote {TARBALL}  ({size_mb:.2f} MB; arXiv limit is 50 MB)")
    print("\nNext: test locally with  cd arxiv_submission && latexmk -pdf "
          f"{MAIN_TEX}\n      or upload the tarball directly to arXiv.")


if __name__ == "__main__":
    main()
