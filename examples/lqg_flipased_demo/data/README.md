# FliPASED model files

The FliPASED P-FLEX model is not redistributed in this repository. It comes from the
FliPASED project dataverse hosted by SZTAKI: https://repo.researchdata.hu/dataverse/flipased

The scripts expect these files in this folder:

- `flipased_modal_sym.npz` (symmetric channel)
- `flipased_modal_asym.npz` (asymmetric channel)
- `flipased_nominal.mat` (nominal model)

Each `.npz` file holds the modal-form state-space model on the 11-point airspeed grid
(`A_modal`, `B_modal`, `C_modal`, `D_modal`, `V_grid`) together with the modal block
indices (`mode_starts`, `mode_ends`, `mode_kinds`, `mode_freqs_hz`, `mode_zetas`,
`scale_factors`, `T_modal`, `T_modal_inv`). See `../dynamics.py` for how they are loaded.

For the preprocessed files used in the paper, please contact the authors.
