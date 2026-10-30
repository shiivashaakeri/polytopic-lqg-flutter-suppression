# LQG Flutter Suppression on Polytopic TP Models via Policy Optimization

Code for the AIAA SciTech paper *LQG Flutter Suppression on Polytopic TP Models via Policy
Optimization* by Shiva Shakeri, Péter Baranyi, and Mehran Mesbahi.

We synthesize polytopic LQG controllers for active flutter suppression on the FliPASED
P-FLEX demonstrator, modeled as a polytopic LPV plant in tensor product (TP) form. The
state-feedback vertex gains are found by Riemannian gradient descent on the polytopic
stabilizing set, with a closed-form polytopic stability certificate as the step-size bound,
and the observer gains come from the same primal procedure by Kalman duality. The method is
compared against Tanaka-Wang PDC LMI synthesis on three TP convex hulls (CNO, IRNO, SNNN).

## Repository layout

```
tptool/                       Python port of the TP Tool toolbox
  lpv/                        sampling, HOSVD, TP transformation, discretization
  hull/                       convex hull manipulation (CNO, IRNO, INO, SNNN, CLOSE, BOX)
  lmi/                        PDC LMI synthesis (baseline)
  prhpg/                      polytopic policy-gradient synthesis and observer utilities
  util/                       control and plotting helpers
examples/lqg_flipased_demo/   FliPASED plant model and the paper experiments
  dynamics.py                 FliPASED LPV model with modal-block reduction
  pipeline.py                 synthesis and closed-loop simulation pipeline
  experiment_*.py             experiments (see below)
  data/                       place the FliPASED model files here (not included)
```

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

The FliPASED model files are not included. See
[`examples/lqg_flipased_demo/data/README.md`](examples/lqg_flipased_demo/data/README.md).

## Reproducing the paper

Run each script from the repository root as a module, for example:

```bash
python -m examples.lqg_flipased_demo.experiment_tp_weights
```

| Result in the paper | Script |
|-|-|
| Fig. 2: TP weighting functions for each hull | `experiment_tp_weights.py` |
| Figs. 3 and 7: open-loop eigenvalues and closed-loop pole migration | `experiment_pole_migration.py` |
| Figs. 5 and 6: synthesis convergence and gradient-dominance check, K side | `experiment_p_riemannian_grad.py` |
| Figs. 5 and 6: observer side (by duality) | `experiment_p_riemannian_dual.py` |
| Joint LQG synthesis | `experiment_p_qrnpo_lqg.py` |
| Table 4: hull sweep against the LMI baseline | `experiment_pqrnpo_hull_invariance.py` |
| Fig. 8: time-domain airspeed sweep with sensor noise and Dryden turbulence | `experiment_turbulence.py` |
| Fig. 9: disturbance-to-flutter-mode frequency response | `experiment_pqrnpo_disturbance_rejection.py` |

Scripts write their outputs to `results/`, `figures/`, or `paper_figures/` subfolders.

### Additional experiments

- Method variants: `experiment_p_qrnpo.py` (quasi-Riemannian Newton, primal),
  `experiment_p_qrnpo_dual.py` (observer), `experiment_p_riemannian_newton.py`
  (full Riemannian Newton).
- Further closed-loop studies with the same controllers:
  `experiment_pqrnpo_flutter_envelope.py`, `experiment_pqrnpo_initial_condition.py`,
  `experiment_pqrnpo_airspeed_transient.py`, `experiment_pqrnpo_robustness_margins.py`,
  `experiment_pqrnpo_uncertainty.py` (re-render its figure with
  `rescue_uncertainty_figure.py`), `experiment_pqrnpo_saturation.py`,
  `experiment_pqrnpo_online_cost.py`.
- Sanity checks: `smoke_test_lqg.py`, `diagnose_lmi.py`.

## Acknowledgments

`tptool` is a Python port of the
[TP Tool MATLAB toolbox](https://www.mathworks.com/matlabcentral/fileexchange/25514-tp-tool)
by Péter Baranyi and the TP Tool team. The FliPASED model is from the FliPASED project
(SZTAKI).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
