"""Plot the measured ITO current-voltage sweeps as scatter data."""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_intensity_response import load_channel_means


if __name__ == "__main__":
    data_dir = Path("ito_light_response/data/ito_characterization_1/voltage_sweep")  # Read-only voltage-sweep CSV files.
    output_path = Path("ito_light_response/output/voltage_response.png")  # Saved current-versus-voltage scatter plot.
    shunt_resistance_ohm = 1000.0  # Resistance used to convert the measured shunt voltage into current.
    amplifier_gain = 1000.0  # Voltage gain between the shunt and oscilloscope CH2.
    sweep_files = ("on900s.csv", "off900s.csv")  # Stabilized voltage sweeps acquired after 900 seconds.
    sweep_labels = ("light on after 900 s", "light off after 900 s")  # Legend labels for the two retained conditions.
    sweep_colors = ("#2166ac", "#b2182b")  # Blue and red distinguish the illuminated and dark conditions.
    marker_size = 24  # Area of each scatter marker in points squared.
    marker_alpha = 0.75  # Opacity used to keep overlapping measurements visible.
    figure_size = (8, 6.5)  # Width and height of the saved figure in inches.
    output_dpi = 300  # Resolution of the saved figure in dots per inch.

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)

    for sweep_file, sweep_label, sweep_color in zip(sweep_files, sweep_labels, sweep_colors):
        applied_voltages, amplifier_voltages = load_channel_means(data_dir / sweep_file)
        measured_currents = amplifier_voltages / (shunt_resistance_ohm * amplifier_gain) * 1e6
        axis.scatter(
            applied_voltages,
            measured_currents,
            s=marker_size,
            color=sweep_color,
            alpha=marker_alpha,
            edgecolors="none",
            label=sweep_label,
        )
        print(
            f"{sweep_file}: {len(applied_voltages)} points, voltage={applied_voltages.min():.4f} to "
            f"{applied_voltages.max():.4f} V, current={measured_currents.min():.5f} to {measured_currents.max():.5f} uA."
        )

    axis.axhline(0.0, color="0.70", linewidth=1.2, linestyle="--", zorder=0)
    axis.axvline(0.0, color="0.70", linewidth=1.2, linestyle="--", zorder=0)
    axis.set_title("ITO current-voltage response after 900 s")
    axis.set_xlabel("Applied voltage (V)")
    axis.set_ylabel(r"Measured current $I_{\mathrm{cur}}$  ($\mu$A)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")
    figure.savefig(output_path, dpi=output_dpi)
    plt.close(figure)
    print(f"Saved voltage-response figure to {output_path}.")
