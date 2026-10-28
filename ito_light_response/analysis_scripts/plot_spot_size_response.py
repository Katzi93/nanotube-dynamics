"""Plot the light-induced ITO current against integrated illumination for the spot-size sweep."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from plot_intensity_response import load_channel_means, measurement_number, paired_current_changes


def linear_response(illumination_intensity, response_coefficient):
    """Evaluate the origin-constrained linear light response."""
    return response_coefficient * illumination_intensity


def fit_linear_response(illumination_intensities, current_changes):
    """Fit an unweighted straight line constrained to pass through the origin."""
    squared_intensity_sum = np.sum(illumination_intensities**2)
    response_coefficient = np.sum(illumination_intensities * current_changes) / squared_intensity_sum
    fitted_currents = linear_response(illumination_intensities, response_coefficient)
    residual_sum_squares = np.sum((current_changes - fitted_currents) ** 2)
    residual_variance = residual_sum_squares / (len(current_changes) - 1)
    response_coefficient_error = np.sqrt(residual_variance / squared_intensity_sum)
    total_sum_squares = np.sum((current_changes - current_changes.mean()) ** 2)
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return response_coefficient, response_coefficient_error, r_squared


if __name__ == "__main__":
    data_dir = Path("ito_light_response/data/ito_characterization_1/spot_sizes")  # Read-only spot-size sweep inputs.
    measurement_dir = data_dir / "current_measurments"  # Oscilloscope CSV files with alternating off/on measurements.
    image_dir = data_dir / "images"  # Camera images whose folder names match the CSV measurement names.
    background_path = Path("ito_light_response/data/ito_characterization_1/intensity_sweep/images/back/Pos0/img_000000000_Default0_000.tif")  # Shared camera background acquired with the same settings.
    output_path = Path("ito_light_response/output/spot_size_response.png")  # Saved current-versus-integrated-illumination plot.
    shunt_resistance_ohm = 1000.0  # Resistance used to convert the measured shunt voltage into current.
    amplifier_gain = 1000.0  # Voltage gain between the shunt and oscilloscope CH2.
    intensity_scale = 1e9  # Scale integrated camera counts to readable x-axis values.
    maximum_intensity = 12.0  # Exclude inconsistent measurements above this integrated intensity in scaled units.
    measurement_color = "#2166ac"  # Blue used for primary measurements in the repository's analysis plots.
    figure_size = (8, 6.5)  # Width and height of the saved figure in inches.
    output_dpi = 300  # Resolution of the saved figure in dots per inch.

    background = tifffile.imread(background_path).astype(np.float64)
    measurement_paths = sorted(measurement_dir.glob("*.csv"), key=measurement_number)
    image_paths = {}
    for image_set_dir in image_dir.iterdir():
        if image_set_dir.is_dir():
            image_path, = image_set_dir.rglob("*.tif")
            image_paths[image_set_dir.name] = image_path

    records = []
    all_applied_voltages = []
    for measurement_path in measurement_paths:
        dataset_name = measurement_path.stem
        if dataset_name not in image_paths:
            print(f"Excluded {measurement_path.name}: no matching image.")
            continue

        image = tifffile.imread(image_paths[dataset_name]).astype(np.float64)
        integrated_intensity = np.sum(image - background) / intensity_scale
        if integrated_intensity > maximum_intensity:
            print(
                f"Excluded {measurement_path.name}: I_int={integrated_intensity:.4f} x 10^9 a.u. "
                f"is above the {maximum_intensity:g} x 10^9 a.u. cutoff."
            )
            continue

        ch1_means, ch2_means = load_channel_means(measurement_path)
        current_deltas = paired_current_changes(ch2_means, shunt_resistance_ohm, amplifier_gain)
        mean_current_delta = current_deltas.mean()
        current_delta_sem = current_deltas.std(ddof=1) / np.sqrt(len(current_deltas))
        records.append((dataset_name, integrated_intensity, mean_current_delta, current_delta_sem))
        all_applied_voltages.extend(ch1_means)

    records.sort(key=lambda record: record[1])
    dataset_names = [record[0] for record in records]
    illumination_intensities = np.array([record[1] for record in records])
    current_changes = np.array([record[2] for record in records])
    current_errors = np.array([record[3] for record in records])
    applied_voltages = np.array(all_applied_voltages)

    response_coefficient, response_coefficient_error, fit_r_squared = fit_linear_response(illumination_intensities, current_changes)
    minimum_applied_voltage = applied_voltages.min()
    maximum_applied_voltage = applied_voltages.max()

    for dataset_name, illumination_intensity, current_change, current_error in zip(
        dataset_names, illumination_intensities, current_changes, current_errors
    ):
        print(
            f"{dataset_name}: I_int={illumination_intensity:.4f} x 10^9 a.u., "
            f"delta I_cur={current_change:.5f} +/- {current_error:.5f} uA"
        )
    print(f"Applied CH1 voltage range: {minimum_applied_voltage:.6f} to {maximum_applied_voltage:.6f} V.")
    print(
        f"Origin-constrained linear fit: c={response_coefficient:.6f} +/- {response_coefficient_error:.6f} "
        f"uA/(10^9 a.u.), R^2={fit_r_squared:.4f}."
    )

    fit_intensities = np.linspace(0.0, illumination_intensities.max() * 1.05, 300)
    fit_currents = linear_response(fit_intensities, response_coefficient)
    fit_label = rf"linear fit: $\Delta I_{{\mathrm{{cur}}}}=cI_{{\mathrm{{int}}}}$, $c={response_coefficient:.4f}$  ($R^2={fit_r_squared:.3f}$)"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    measurement_handle = axis.errorbar(
        illumination_intensities,
        current_changes,
        yerr=current_errors,
        color=measurement_color,
        marker="o",
        markersize=8,
        linestyle="none",
        capsize=4,
        label=r"measurements (mean $\pm$ SEM)",
    )
    fit_handle, = axis.plot(fit_intensities, fit_currents, color="black", linewidth=3.0, label=fit_label)
    axis.set_title("ITO spot-size response")
    axis.set_xlabel(r"Integrated spot illumination $I_{\mathrm{int}}$  ($10^9$ a.u.)")
    axis.set_ylabel(r"Light-induced current $\Delta I_{\mathrm{cur}}$  ($\mu$A)")
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(handles=[measurement_handle, fit_handle], frameon=False, loc="best")
    figure.savefig(output_path, dpi=output_dpi)
    plt.close(figure)
    print(f"Saved spot-size-response figure to {output_path}.")
