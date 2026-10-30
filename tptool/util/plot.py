"""
Hull / weighting function visualization.

Ported from MATLAB TP Tool: util/plothull.m.
"""

import matplotlib.pyplot as plt
import numpy as np


def plothull(U, domain=None, ax=None, **plot_kwargs):
    """
    Plot weighting functions as line plots.

    Parameters
    ----------
    U : ndarray, list of ndarrays, or list of lists
        Weighting function data. If a list (or list-of-lists) of arrays,
        produces a separate figure per element. If a single 2D array,
        produces one plot.
    domain : array_like, shape (P, 2), optional
        Per-parameter x-axis bounds. If None, uses sample indices on x-axis.
    ax : matplotlib Axes, optional
        For single-array inputs, plot into the given axes instead of a new
        figure. Ignored when U is a list of arrays.
    **plot_kwargs : dict
        Extra keyword arguments forwarded to ``ax.plot`` (e.g. linewidth).

    Returns
    -------
    figs : matplotlib.figure.Figure or list of Figures
        Figure(s) created. If `ax` is provided for a single-array input,
        returns the axes' figure.

    Notes
    -----
    Mirrors the MATLAB plothull. The MATLAB version always opens a new
    figure window per array; we do the same when no `ax` is given.
    """

    # List input -> one figure per element
    if isinstance(U, (list, tuple)):
        figs = []
        for i, U_i in enumerate(U):
            U_i = np.asarray(U_i)  # noqa: PLW2901
            if U_i.size == 0:
                continue
            fig, ax_i = plt.subplots()
            x = _xaxis(U_i, domain[i] if domain is not None else None)
            ax_i.plot(x, U_i, linewidth=plot_kwargs.pop("linewidth", 2), **plot_kwargs)
            ax_i.set_title(f"Weighting functions for parameter {i + 1}")
            ax_i.set_xlabel("p" if domain is not None else "sample index")
            ax_i.set_ylabel("weight")
            ax_i.autoscale(tight=True)
            figs.append(fig)
        return figs

    # Single array
    U = np.asarray(U)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    domain_row = None
    if domain is not None:
        domain_arr = np.asarray(domain)
        # Either (1, 2) or just (2,)
        domain_row = domain_arr.ravel() if domain_arr.size == 2 else domain_arr[0]
    x = _xaxis(U, domain_row)

    ax.plot(x, U, linewidth=plot_kwargs.pop("linewidth", 2), **plot_kwargs)
    ax.autoscale(tight=True)
    ax.set_xlabel("p" if domain is not None else "sample index")
    ax.set_ylabel("weight")
    return fig


def _xaxis(U_i, domain_row):
    """Build x-axis values for one weighting-function array."""
    n_samples = U_i.shape[0]
    if domain_row is None:
        return np.arange(1, n_samples + 1)
    return np.linspace(float(domain_row[0]), float(domain_row[1]), n_samples)
