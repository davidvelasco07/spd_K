# Testing

## Unit tests

Transform kernel correctness (cv ↔ sp sweeps, face integrals):

```bash
./build/spd_K_test
```

## Regression suite

```bash
python tests/run_tests.py --build-dir build
```

Checks include L1 error vs the analytic sine wave, mass conservation to
round-off, golden-file comparison, and implosion mass conservation with
reflective BCs.

Options:

- `--skip-unit` — skip `spd_K_test`
- `--regen-goldens` — refresh reference output files

## Visual suite

Generates PNG panels and `docs/gallery.md` for browser review and Sphinx:

```bash
python tests/visual_suite.py --build-dir build
```

| Flag | Effect |
|---|---|
| `--only PATTERN` | Run cases matching a glob (`--only implosion`) |
| `--fast` | Coarse grid, short runs; skips knob section |
| `--skip-run` | Re-render plots from existing `build/visual_out/` |

Output: `docs/gallery/*.png` and `docs/gallery.md`.

## Performance benchmark

```bash
python tests/benchmark.py --binary build/spd_K
```

Prints zone-cycles/s for RK3 sine-wave sweeps (outputs disabled via
`output/dt=-1`).

## Building the documentation locally

```bash
pip install -r docs/requirements.txt
make -C docs html
# open docs/_build/html/index.html
```

The site is also published to GitHub Pages on push to `main` (see
`.github/workflows/docs.yml`).
