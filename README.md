# surj-repro

A reimplementation, from the paper alone, of **Surj** — Yang & Howe, *Ontological Learning for
Fast, Accurate, and Robust Hierarchical Multi-label Classification* (WWW'22 companion) — whose
code was announced but never released.

Everything is in [`surj_repro/`](surj_repro/README.md): the code, the recipes, the per-seed
result JSONs, and the write-ups [`results.md`](surj_repro/results.md) and
[`analysis.md`](surj_repro/analysis.md).

## Quick start

```bash
git clone https://github.com/EGiunchiglia/C-HMCNN.git
git -C C-HMCNN checkout e887855ab4b3b8f22e4fafc80a0a33c1d1ac0041   # unzip its HMC_data archive if zipped
uv sync
uv run pytest surj_repro/tests -q
surj_repro/run_all.sh sanity ladder hpsearch final extras summary
```

Run every command from the repository root: C-HMCNN's parser reads `./C-HMCNN/HMC_data`.

## Provenance

The code was developed inside [hmc-torch](https://github.com/Sette/hmc-torch) and extracted
from its commit `d19e30f`. The `commit` field stored in each result JSON refers to the
hmc-torch history of that time, not to this repository. Three files were adapted so the folder
runs without the `hmc` package:

- `chmcnn_loader.py` is the subset of hmc-torch's `experiments/compare_chmcnn_loader.py` that
  the reproduction uses;
- `sparse_r.py` and `hierarchy_violation.py` are unmodified copies of
  `hmc/models/hierarchical/sparse_r.py` and `hmc/utils/metrics/hierarchy_violation.py`.

The first column of "parity run" in `results.md` came from hmc-torch's parity script
(`output/chmcnn_parity/results.json`); `summarize.py` leaves it empty when that file is absent.

## License

MIT (see `LICENSE`), except for the parts of `surj_repro/chmcnn_loader.py` marked as verbatim
from `C-HMCNN/main.py` (`their_r`, `get_constr_out`, `ConstrainedFFNNModel`), which come from
[C-HMCNN](https://github.com/EGiunchiglia/C-HMCNN) and remain under its GPL-3.0 license.
C-HMCNN itself is not redistributed here; clone it as shown above.
