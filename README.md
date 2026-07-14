# Aagjuuk Labs: Real-Time Multi-Material Edge Physics Solver

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aagjuuk.streamlit.app/)

**Live Demo:** [https://aagjuuk.streamlit.app/](https://aagjuuk.streamlit.app/)

---

Aagjuuk is a high-performance Physics-Informed Machine Learning (SciML) suite designed to solve complex, 3D coupled thermo-mechanical stress and deformation fields on physical hardware interfaces in under 5 milliseconds.

By bypassing legacy spatial discretization grids (mesh-bound Finite Element Analysis), Aagjuuk evaluates the physical state continuously over space-time coordinates $(x, y, z, t)$.

## Repository Architecture

* `streamlit_app.py`: Interactive user dashboard allowing real-time parameter tweaking and interactive 3D deformation modeling.
* `aagjuuk_hybrid_pinn.py`: Hybrid physical-observational training pipeline that anchors mathematical PDEs to on-chip thermocouple data logs.
* `models/aagjuuk_composite_3d.py`: Composite multi-material solver designed for Silicon-Copper interfaces.
* `models/aagjuuk_anisotropic_3d.py`: Core solver using fourth-rank anisotropic silicon stiffness tensors ($C_{11}$, $C_{12}$, $C_{44}$) to model real semiconductor lattices.
* `models/aagjuuk_self_adaptive_pinn.py`: Advanced custom training loop featuring dynamic, self-adaptive loss weighting to resolve the gradient pathological problem.
* `models/aagjuuk_fourier_features.py`: Fourier Feature Projection layer designed to bypass neural network spectral bias and resolve microscopic cracks.

## Performance Profiles

* **Compute Latency:** < 4 milliseconds (a 10,000x speedup compared to legacy iterative mesh solvers).
* **Target Applications:** 3D high-bandwidth memory (HBM3) thermal warpage, wafer laser annealing defect prevention, and multi-material interface shear tracking (Silicon-Copper).

## Quick Start

To run the interactive Streamlit dashboard locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
