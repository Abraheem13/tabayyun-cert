# TIFS manuscript

- `main.tex`: manuscript (IEEEtran journal, 10 pages). All figures are TikZ/pgfplots, black and white.
- `refs.bib`: references. Each entry was checked against Crossref or the arXiv API on 2026-11-17.
- `data/`: every table body (`tab_*.tex`) and plot data file (`*.dat`), generated from `results/` by
  `python scripts/paper_numbers.py`. `data/numbers.json` holds every other number quoted in the text.

Build:

    python scripts/paper_numbers.py
    cd paper/tifs && latexmk -pdf main.tex
