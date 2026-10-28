"""Compare the illuminated and dark ITO currents after 900 seconds."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_intensity_response import load_channel_means


def complete_sweep_branches(applied_voltages, measured_currents):
    """Return the complete rising and falling branches bounded by the global voltage extrema."""
    maximum_index = np.argmax(applied_voltages)
    rising_start_index = np.argmin(applied_voltages[:maximum_index + 1])
    falling_end_index = maximum_index + np.argmin(applied_voltages[maximum_index:])
    rising_branch = (applied_voltages[rising_start_index:maximum_index + 1], measured_currents[rising_start_index:maximum_index + 1])
    falling_branch = (applied_voltages[maximum_index:falling_end_index + 1], measured_currents[maximum_index:falling_end_index + 1])
    return rising_branch, falling_branch


def interpolate_current_difference(on_branch, off_branch):
    """Interpolate the off current onto on-branch voltages within the shared voltage range."""
    on_voltages, on_currents = on_branch
    off_voltages, off_currents = off_branch
    off_order = np.argsort(off_voltages)
    sorted_off_voltages = off_voltages[off_order]
    sorted_off_currents = off_currents[off_order]
    shared_voltage_mask = (on_voltages >= sorted_off_voltages.min()) & (on_voltages <= sorted_off_voltages.max())
    comparison_voltages = on_voltages[shared_voltage_mask]
    interpolated_off_currents = np.interp(comparison_voltages, sorted_off_voltages, sorted_off_currents)
    current_difference = on_currents[shared_voltage_mask] - interpolated_off_currents
    return comparison_voltages, current_difference


if __name__ == "__main__":
    data_dir = Path("ito_light_response/data/ito_characterization_1/voltage_sweep")  # Read-only voltage-sweep CSV files.
    on_path = data_dir / "on900s.csv"  # Illuminated voltage sweep acquired after 900 seconds.
    off_path = data_dir / "off900s.csv"  # Dark voltage sweep acquired after 900 seconds.
    output_path = Path("ito_light_response/output/voltage_response_difference.png")  # Saved on-minus-off current plot.
    shunt_resistance_ohm = 1000.0  # Resistance used to convert the measured shunt voltage into current.
    amplifier_gain = 1000.0  # Voltage gain between the shunt and oscilloscope CH2.
    branch_labels = ("rising voltage", "falling voltage")  # Labels for the complete zigzag branches.
    branch_colors = ("#2166ac", "#b2182b")  # Blue and red distinguish the two sweep directions.
    marker_size = 28  # Area of each scatter marker in points squared.
    marker_alpha = 0.80  # Opacity used to keep overlapping measurements visible.
    figure_size = (8, 6.5)  # Width and height of the saved figure in inches.
    output_dpi = 300  # Resolution of the saved figure in dots per inch.

    on_voltages, on_amplifier_voltages = load_channel_means(on_path)
    off_voltages, off_amplifier_voltages = load_channel_means(off_path)
    on_currents = on_amplifier_voltages / (shunt_resistance_ohm * amplifier_gain) * 1e6
    off_currents = off_amplifier_voltages / (shunt_resistance_ohm * amplifier_gain) * 1e6
    on_branches = complete_sweep_branches(on_voltages, on_currents)
    off_branches = complete_sweep_branches(off_voltages, off_currents)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    for branch_label, branch_color, on_branch, off_branch in zip(branch_labels, branch_colors, on_branches, off_branches):
        comparison_voltages, current_difference = interpolate_current_difference(on_branch, off_branch)
        axis.scatter(
            comparison_voltages,
            current_difference,
            s=marker_size,
            color=branch_color,
            alpha=marker_alpha,
            edgecolors="none",
            label=branch_label,
        )
        print(
            f"{branch_label}: {len(comparison_voltages)} points, voltage={comparison_voltages.min():.4f} to "
            f"{comparison_voltages.max():.4f} V, delta current={current_difference.min():.5f} to "
            f"{current_difference.max():.5f} uA."
        )

    axis.axhline(0.0, color="0.70", linewidth=1.2, linestyle="--", zorder=0)
    axis.axvline(0.0, color="0.70", linewidth=1.2, linestyle="--", zorder=0)
    axis.set_title("ITO light-induced current after 900 s")
    axis.set_xlabel("Applied voltage (V)")
    axis.set_ylabel(r"On-minus-off current $\Delta I_{\mathrm{cur}}$  ($\mu$A)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")
    figure.savefig(output_path, dpi=output_dpi)
    plt.close(figure)
    print(f"Saved voltage-response-difference figure to {output_path}.")
