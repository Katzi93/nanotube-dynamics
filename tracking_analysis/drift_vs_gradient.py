"""Fit the along-tube drift velocity against the projected light gradient and plot the perpendicular control.

Each tube contributes one bare central-difference velocity split along its orientation axis t into v_parallel and
v_perp, and one projected light gradient ∇I·t with the axis oriented toward the brighter tip so both are nonnegative in
the driving direction. If the tubes drift up the illumination gradient, v_parallel rises with the projected gradient
through the origin with slope equal to the mobility, while v_perp stays scattered about zero with no gradient dependence.

All quantities are reported in physical units: velocities in um/s, the projected gradient in a.u./um, and the angular
velocity in 1/s, using the calibration frame_time = 0.5 s and 2048 px = 221.66666 um.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tracking_analysis.tube_tracks import (
    add_light_gradient,
    add_projections,
    add_velocity,
    build_tube_tracks,
    load_datasets,
    simplify_dataframe,
)

# --- Physical calibration -------------------------------------------------------------------------------------------
FRAME_TIME_S = 0.5  # Seconds between successive frames.
MICRONS_PER_PIXEL = 221.66666 / 2048.0  # 2048 px span 221.66666 um in the imaging plane.
PIXELS_PER_FRAME_TO_UM_PER_S = MICRONS_PER_PIXEL / FRAME_TIME_S  # Convert a pixels/frame velocity to um/s.
INTENSITY_PER_PIXEL_TO_PER_UM = 1.0 / MICRONS_PER_PIXEL  # Convert an intensity/pixel gradient to a.u./um.
RAD_PER_FRAME_TO_PER_S = 1.0 / FRAME_TIME_S  # Convert a radians/frame angular velocity to 1/s.

# --- Publication styling --------------------------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "figure.facecolor": "white",
})
DATASET_LABELS = {"test_1": "Dataset 1", "test_2": "Dataset 2", "test_3": "Dataset 3"}  # Publication legend labels.
UNIT_MU_TRANSLATION = r"$\mu\mathrm{m}^2\,\mathrm{s}^{-1}\,\mathrm{a.u.}^{-1}$"  # Units of a translational mobility.
UNIT_MU_ROTATION = r"$\mathrm{rad}\,\mu\mathrm{m}\,\mathrm{s}^{-1}\,\mathrm{a.u.}^{-1}$"  # Units of the rotational mobility.
GRADIENT_LABEL = r"$\nabla I \cdot \hat{t}$  (a.u. $\mu$m$^{-1}$)"  # Shared, compact x-axis label.


def fit_through_origin(gradient, velocity):
    """Return the through-origin slope and its coefficient of determination around zero for velocity = slope * gradient."""
    slope = np.dot(gradient, velocity) / np.dot(gradient, gradient)
    residuals = velocity - slope * gradient
    r_squared = 1.0 - np.dot(residuals, residuals) / np.dot(velocity, velocity)
    return slope, r_squared


def style_axis(axis):
    """Strip the top and right spines so each panel reads as a clean publication plot."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


if __name__ == "__main__":
    dataset_names = ["test_1", "test_2", "test_3"]  # Canonical track datasets pooled from tracking_data/.
    background_path = Path("test_data/background.tif")  # Read-only illumination profile shared by every movie.
    output_path = Path("analysis_outputs/drift_vs_gradient.png")  # Figure written by this analysis.
    dataset_colors = {"test_1": "#2166ac", "test_2": "#b2182b", "test_3": "#1b7837"}  # One color per pooled movie.
    figure_size = (18, 5.5)  # Width and height of the saved figure in inches.
    output_dpi = 300  # Resolution of the saved figure in dots per inch.

    ridges = simplify_dataframe(load_datasets(dataset_names))
    tube_tracks = build_tube_tracks(ridges)
    tube_tracks = add_velocity(tube_tracks)
    tube_tracks = add_light_gradient(tube_tracks, background_path)
    tube_tracks = add_projections(tube_tracks)
    measured = tube_tracks.dropna(subset=["v_parallel", "v_perp", "angular_velocity"])

    # Convert every fitted quantity from tracker units (pixels, frames) into physical units before plotting or fitting.
    projected_gradient = measured["projected_gradient"].to_numpy() * INTENSITY_PER_PIXEL_TO_PER_UM
    v_parallel = measured["v_parallel"].to_numpy() * PIXELS_PER_FRAME_TO_UM_PER_S
    v_perp = measured["v_perp"].to_numpy() * PIXELS_PER_FRAME_TO_UM_PER_S
    angular_velocity = measured["angular_velocity"].to_numpy() * RAD_PER_FRAME_TO_PER_S
    mu_parallel, mu_parallel_r_squared = fit_through_origin(projected_gradient, v_parallel)
    mu_perp, mu_perp_r_squared = fit_through_origin(projected_gradient, v_perp)
    mu_rot, mu_rot_r_squared = fit_through_origin(projected_gradient, angular_velocity)
    fraction_up_gradient = np.mean(v_parallel > 0.0)
    print(f"Fitting {len(measured)} central-difference velocities from {measured.groupby(['dataset', 'track']).ngroups} tracks.")
    print(f"Parallel mobility mu_par={mu_parallel:.4f} um^2/s per a.u./um, through-origin R^2={mu_parallel_r_squared:.3f}.")
    print(f"Fraction of tubes moving up-gradient (v_parallel > 0)={fraction_up_gradient:.3f}.")
    print(f"Perpendicular mobility mu_perp={mu_perp:.4f} um^2/s per a.u./um, through-origin R^2={mu_perp_r_squared:.3f}.")
    print(f"Rotational mobility mu_rot={mu_rot:.5f} rad*um/s per a.u./um, through-origin R^2={mu_rot_r_squared:.3f}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True, sharex=True)
    for dataset_name, color in dataset_colors.items():
        dataset_rows = (measured["dataset"] == dataset_name).to_numpy()
        label = DATASET_LABELS[dataset_name]
        axes[0].scatter(projected_gradient[dataset_rows], v_parallel[dataset_rows], color=color, s=32, alpha=0.55, edgecolor="none", label=label)
        axes[1].scatter(projected_gradient[dataset_rows], v_perp[dataset_rows], color=color, s=32, alpha=0.55, edgecolor="none", label=label)
        axes[2].scatter(projected_gradient[dataset_rows], angular_velocity[dataset_rows], color=color, s=32, alpha=0.55, edgecolor="none", label=label)

    fit_gradient = np.array([0.0, projected_gradient.max()])
    for axis in axes:
        axis.axhline(0.0, color="0.6", linewidth=1.2, zorder=0)
        style_axis(axis)

    shared_velocity_ylim = (-3.0, 5.5)  # Common y-range shared by the parallel and perpendicular velocity panels.

    # Panel 1: parallel drift, the driven channel. Its through-origin slope is the parallel mobility mu_||.
    axes[0].plot(fit_gradient, mu_parallel * fit_gradient, color="black", linewidth=3.0, label="fit function")
    axes[0].set_title("Parallel drift" + "\n" + rf"$\mu_\parallel = {mu_parallel:.2g}$ {UNIT_MU_TRANSLATION}")
    axes[0].set_xlabel(GRADIENT_LABEL)
    axes[0].set_ylabel(r"$v_\parallel$  ($\mu$m s$^{-1}$)")
    axes[0].set_ylim(shared_velocity_ylim)
    axes[0].legend(frameon=True, loc="upper right")

    # Panel 2: perpendicular control, expected to sit near zero with a negligible slope mu_perp.
    axes[1].plot(fit_gradient, mu_perp * fit_gradient, color="black", linewidth=3.0, label="fit function")
    axes[1].set_title("Perpendicular drift" + "\n" + rf"$\mu_\perp = {mu_perp:.2g}$ {UNIT_MU_TRANSLATION}")
    axes[1].set_xlabel(GRADIENT_LABEL)
    axes[1].set_ylabel(r"$v_\perp$  ($\mu$m s$^{-1}$)")
    axes[1].set_ylim(shared_velocity_ylim)
    axes[1].legend(frameon=True, loc="upper right")

    # Panel 3: rotational control, the axial angular velocity against the same gradient with rotational mobility mu_rot.
    axes[2].plot(fit_gradient, mu_rot * fit_gradient, color="black", linewidth=3.0, label="fit function")
    axes[2].set_title("Rotational drift" + "\n" + rf"$\mu_\mathrm{{rot}} = {mu_rot:.2g}$ {UNIT_MU_ROTATION}")
    axes[2].set_xlabel(GRADIENT_LABEL)
    axes[2].set_ylabel(r"$\omega$  (s$^{-1}$)")
    axes[2].legend(frameon=True, loc="upper right")

    figure.savefig(output_path, dpi=output_dpi)
    plt.close(figure)
    print(f"Saved drift-versus-gradient figure to {output_path}.")
