"""Plot the light-induced ITO current and fit a saturating response."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import tifffile


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


def measurement_number(path):
    """Return the numeric part of a measurement filename such as 4nd.csv."""
    return int("".join(character for character in path.stem if character.isdigit()))


def load_channel_means(csv_path):
    """Load the recorded mean voltage for CH1 and CH2 in measurement order."""
    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    ch1_means = np.array([float(row["Mean"]) for row in rows if row["Channel"] == "CH1"])
    ch2_means = np.array([float(row["Mean"]) for row in rows if row["Channel"] == "CH2"])
    return ch1_means, ch2_means


def paired_current_changes(ch2_means, shunt_resistance_ohm, amplifier_gain):
    """Convert successive off/on CH2 voltage pairs into light-induced currents in microamperes."""
    delta_output_voltage = ch2_means[1::2] - ch2_means[0::2]
    return delta_output_voltage / (shunt_resistance_ohm * amplifier_gain) * 1e6


def saturating_response(illumination_intensity, saturation_current, half_saturation_intensity):
    """Evaluate a saturating hyperbola for the light-induced current."""
    return saturation_current * illumination_intensity / (half_saturation_intensity + illumination_intensity)


def fit_response(response_function, illumination_intensities, current_changes, current_errors, initial_parameters):
    """Fit a response function using the current standard errors as absolute uncertainties."""
    parameters, covariance = curve_fit(
        response_function,
        illumination_intensities,
        current_changes,
        sigma=current_errors,
        absolute_sigma=True,
        p0=initial_parameters,
        bounds=(0.0, np.inf),
    )
    fitted_current = response_function(illumination_intensities, *parameters)
    residual_sum_squares = np.sum((current_changes - fitted_current) ** 2)
    total_sum_squares = np.sum((current_changes - current_changes.mean()) ** 2)
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return parameters, covariance, r_squared


if __name__ == "__main__":
    data_dir = Path("ito_light_response/data/ito_characterization_1/intensity_sweep")  # Read-only intensity-sweep inputs.
    measurement_dir = data_dir / "current_measurments"  # Oscilloscope CSV files with alternating off/on measurements.
    image_dir = data_dir / "images"  # Camera images whose folder names match the CSV measurement names.
    background_path = image_dir / "back/Pos0/img_000000000_Default0_000.tif"  # Camera background recorded without illumination.
    output_path = Path("ito_light_response/output/intensity_response.png")  # Saved current-versus-illumination plot.
    shunt_resistance_ohm = 1000.0  # Resistance used to convert the measured shunt voltage into current.
    amplifier_gain = 1000.0  # Voltage gain between the shunt and oscilloscope CH2.
    intensity_scale = 1e9  # Scale integrated camera counts to readable x-axis values.
    maximum_intensity = 12.0  # Exclude inconsistent measurements above this integrated intensity in scaled units.
    measurement_color = "#2166ac"  # Blue used for primary measurements in the repository's analysis plots.
    figure_size = (8, 6.5)  # Width and height of the saved figure in inches.
    output_dpi = 300  # Resolution of the saved figure in dots per inch.

    background = tifffile.imread(background_path).astype(np.float64)
    measurement_paths = sorted(measurement_dir.glob("*.csv"), key=measurement_number)
    records = []
    all_applied_voltages = []

    for measurement_path in measurement_paths:
        dataset_name = measurement_path.stem
        image_path, = (image_dir / dataset_name).rglob("*.tif")
        image = tifffile.imread(image_path).astype(np.float64)
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

    saturation_initial_parameters = (current_changes.max(), np.median(illumination_intensities))
    saturation_parameters, saturation_covariance, saturation_r_squared = fit_response(
        saturating_response, illumination_intensities, current_changes, current_errors, saturation_initial_parameters
    )
    saturation_current, half_saturation_intensity = saturation_parameters
    saturation_current_error, half_saturation_intensity_error = np.sqrt(np.diag(saturation_covariance))
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
        f"Saturation fit: I_sat={saturation_current:.5f} +/- {saturation_current_error:.5f} uA, "
        f"I_1/2={half_saturation_intensity:.4f} +/- {half_saturation_intensity_error:.4f} x 10^9 a.u., "
        f"R^2={saturation_r_squared:.4f}."
    )

    fit_intensities = np.linspace(0.0, illumination_intensities.max() * 1.05, 300)
    saturation_fit_currents = saturating_response(fit_intensities, saturation_current, half_saturation_intensity)
    saturation_label = rf"saturation fit: $I_{{\mathrm{{sat}}}}={saturation_current:.3f}$, $I_{{1/2}}={half_saturation_intensity:.2f}$  ($R^2={saturation_r_squared:.3f}$)"

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
    saturation_handle, = axis.plot(
        fit_intensities,
        saturation_fit_currents,
        color="black",
        linewidth=3.0,
        label=saturation_label,
    )
    axis.set_title("ITO light response")
    axis.set_xlabel(r"Integrated illumination $I_{\mathrm{int}}$  ($10^9$ a.u.)")
    axis.set_ylabel(r"Light-induced current $\Delta I_{\mathrm{cur}}$  ($\mu$A)")
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(handles=[measurement_handle, saturation_handle], frameon=False, loc="best")
    figure.savefig(output_path, dpi=output_dpi)
    plt.close(figure)
    print(f"Saved intensity-response figure to {output_path}.")
