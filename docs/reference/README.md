# Reference material

Upstream documentation and source copies for the methods ChronoScope wraps.
**None of this is imported by the application** — it is kept for provenance, so
that each wrapper in `core/` can be checked against the implementation it was
derived from.

| Folder | Contents | Upstream |
|---|---|---|
| `cosinorpy/` | Demo notebooks and terminology notes for CosinorPy, plus a map of which CosinorPy calls back each GUI method. | https://github.com/mmoskon/CosinorPy |
| `circacompare/` | Technical documentation and a Python transcription of the CircaCompare model, used as the reference while writing `core/circacompare_analysis.py`. The transcription is not runnable as-is (`from src import constants`). | https://github.com/RWParsons/circacompare |
| `rhythm_analysis/` | Notes on the classical methods implemented in `core/rhythm_analysis.py` (JTK_CYCLE, Lomb–Scargle, CWT, F24, LME). | — |

Code that ChronoScope actually executes lives in `core/`; third-party code that
is bundled and imported lives in `core/vendor/`.
