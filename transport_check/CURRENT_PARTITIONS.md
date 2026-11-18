# Prescribed measured-current comparison

This test compares two different reaction assignments while allowing individual species concentrations, pH, electric potential, and electroosmotic flow to respond. The measured total current is an input. It is never reduced to obtain a solution.

The first assignment puts every electron into ascorbate oxidation. The second assigns a specified fraction to ascorbate and the remainder to **hypothetical buffered water oxidation**. The latter is a test of model consequences, not identification of an experimental reaction or a fit of the current partition.

## Reaction bookkeeping

Let `f` denote the fraction of the local electron current carried by ascorbate oxidation. Both reactions are assigned the same normalized Gaussian spatial profile. Per two electron equivalents, the reactions are

\[
\mathrm{AscH^-+Tris\longrightarrow DHA+TrisH^++2e^-},
\]

\[
\mathrm{H_2O+2Tris\longrightarrow \tfrac12 O_2+2TrisH^++2e^-}.
\]

With `q = j_s/(2F)`, the mixed microscopic surface sources into the electrolyte are

| Species | Source divided by q |
| --- | ---: |
| AscH− | −f |
| Neutral Tris | −(2−f) |
| TrisH+ | +(2−f) |
| DHA | +f |
| H2O | −(1−f) |
| O2 | +(1−f)/2 |

The aqueous charge source is `2q`, independently of `f`. Total Tris has zero net surface source; its protonated and neutral forms exchange through the local acid-base equilibria. Ascorbate consumption is `f j_s/(2F)`. The zero-flux spectator component conditions, charge conservation, and electroneutrality are retained.

Water is treated as solvent. Oxygen is a neutral, one-way product in this ideal model: its source is recorded, but its passive transport and possible bubble formation are omitted. It does not enter the solved ideal ionic chemistry. The neutral DHA approximation and all other chemical uncertainties of the original [transport check](README.md) remain.

Buffer-assisted water oxidation can be represented by one proton accepted by a buffer per electron. This is the explicit alternative tested here; the experiment does not establish that it occurs at ITO. See the primary models by [Dau and Pasquini (2019)](https://refubium.fu-berlin.de/handle/fub188/23982) and [Wada, Obata, and Takanabe (2026)](https://pubs.rsc.org/en/content/articlehtml/2026/ey/d6ey00006a) for electrolyte-base transport in water oxidation.

## Operating points and assumptions

The two currents taken from the existing analysis near 2 V are:

- **113.8532 nA:** illuminated current minus dark current.
- **172.158 nA:** total illuminated current.

Each is separately assigned to the same Gaussian spot, with half-maximum radius **52.541 µm**. These are alternative localization assumptions. They must not be added together. In particular, this half-space test does not establish that the measured dark current is localized to the illumination spot.

The ascorbate reservoir concentration is **0.5 mM**, reconstructed from the documented 100 mM stock and 0.005 volume fraction. The unfitted chemistry predicts a reservoir conductivity of **377.67 µS/cm**, compared with the measured 435 µS/cm. The freely dissolved OPA amount, its mobility and protonation constants, and the stock reconstruction remain model inputs. The calculated spatial conductivity changes are relative to the model reservoir, not an imposed measured conductivity.

The tested fractions are `f = 0, 0.05, 0.10`. The last two correspond to ascorbate currents of approximately 5.69/11.39 nA at the lower total current and 8.61/17.22 nA at the higher current. A higher total current also changes migration, so the earlier all-ascorbate depletion threshold must not simply be divided by total current to infer an exact upper fraction.

The flow comparison uses `zeta = 0` and `zeta = -2 V_T = -51.385 mV`, with the same Stokes/slip construction as [HYDRODYNAMICS.md](HYDRODYNAMICS.md). No 2 V boundary condition is applied to the outer electrolyte.

## Numerical checks and outputs

```powershell
& 'C:/Users/Flori/anaconda3/envs/tube-tracking-clean/python.exe' transport_check/run_current_partitions.py
& 'C:/Users/Flori/anaconda3/envs/tube-tracking-clean/python.exe' transport_check/validate_current_partitions.py
```

The driver runs 40² and 64² meshes on a domain with radius and height `16a`. The full-current, 5% case also runs on an enlarged `32a` domain with 96² cells. Positive remote boundary states come from a finite, integrated no-flow chemical path with the same current partition. Extrapolation beyond that path is forbidden.

Generated data are in `output/current_partitions/`:

- `summary.json` and `case_summary.csv`: prescribed currents, reaction shares, spatial extrema, surface values, convergence histories, and the ascorbate-only audit.
- `*.npz`: component and individual-species concentrations with labels, electric potential, conductivity, and flow fields.
- `*.png`: concentration, conductivity, potential, and liquid-speed maps.
- `validation.json`: reaction bookkeeping, conservation, positivity, no-flow comparison, and mesh/domain checks.

The additional ascorbate-only audit attempts fixed currents below and above first depletion, including the two measured-current scenarios. Failure to converge with nonnegative concentrations is recorded as a failure to find a physical solution. **Solver failure alone is not a proof that no solution exists.** The separately validated first-depletion branch, the exact no-flow solution, and the prescribed-current attempts are distinct diagnostics.

## Interpretation

The earlier small conductivity change was obtained near first ascorbate depletion at about 15 nA. It cannot be transferred to a different reaction carrying 114–172 nA.

In the mixed-reaction test, protonation changes and migration of the nonreacting components can change conductivity even when ascorbate is only weakly consumed. A buffer limits pH changes according to its finite capacity; it does not require spatially uniform ionic concentrations or conductivity. For `f=0`, ascorbate is itself a spectator and may accumulate by migration.

This comparison tests whether constant conductivity follows from these stated assumptions. It does not determine which reaction occurs experimentally, the fraction of current carried by each reaction, or the finite-cell current distribution.

## Calculated operating points

For the 64² mesh, including flow with `zeta = -51.385 mV`:

| Prescribed total current | Ascorbate current share | Axis surface ascorbate | Axis surface potential | Maximum surface conductivity change |
| --- | ---: | ---: | ---: | ---: |
| 113.853 nA | 0% | 0.6997 mM | 8.710 mV | +47.51% |
| 113.853 nA | 5% | 0.4684 mM | 8.784 mV | +45.62% |
| 113.853 nA | 10% | 0.2365 mM | 8.859 mV | +43.75% |
| 172.158 nA | 0% | 0.8025 mM | 12.314 mV | +69.30% |
| 172.158 nA | 5% | 0.4311 mM | 12.432 mV | +66.74% |
| 172.158 nA | 10% | 0.0580 mM | 12.552 mV | +64.19% |

All these operating points have positive concentrations. The conductivity changes are relative to the unfitted model reservoir, **377.67 µS/cm**. They are numerical predictions under the stated reaction and dissolved-OPA assumptions, not measured spatial changes.

The full-current 5% case has an axis surface pH near **8.24**, compared with **8.82** in the reservoir. Total dissolved OPA rises from **2.5 to about 5.8 mM**, while total spermidine decreases. These changes contribute to the substantial conductivity variation; the calculation does not hold spectator-ion concentrations constant.

Removing flow changes the full-current 5% surface ascorbate from about 0.431 to 0.428 mM and its conductivity increase from about 66.7% to 67.7%. Electroosmotic transport at this prescribed zeta therefore changes the results modestly. It does not suppress the calculated conductivity variation.

At a 10% share of 172.158 nA, ascorbate carries **17.216 nA**, exceeding the approximately 14.7 nA first-depletion threshold of the all-ascorbate model. There is no contradiction: the larger total current produces a stronger electric field, which drives additional ascorbate toward the electrode by migration. The depletion threshold depends on the complete transport problem, not only the partial ascorbate current.

## Ascorbate-only audit and validation

Assigning every electron to ascorbate gives the following fixed-current attempts with flow:

| Prescribed current | Result |
| --- | --- |
| 10 nA | Converged; surface ascorbate 0.1617 mM |
| 14 nA | Converged; surface ascorbate 0.0230 mM |
| 14.6 nA | Converged; surface ascorbate 0.0020 mM |
| 15 and 20 nA | No positive converged solution found |
| 113.853 and 172.158 nA | No positive converged solution found |

These failures are consistent with the separately mesh-validated first-depletion branch at **14.6655 nA** with flow. They are not an imposed input limit or proof that the experimental cell cannot carry a larger current. The all-current-to-fresh-ascorbate assignment, its reservoir concentration, the spatial current assignment, and the transport model are the assumptions being tested.

The reaction-balance, independent no-flow, and coupled-flow validation scripts all pass. In the mixed-current runs, the maximum freshly evaluated scaled transport residual is below **1.5 × 10⁻¹⁰**, and the global component/current imbalance is below **2 × 10⁻¹¹** of the imposed reaction/electron rate. All calculated species concentrations are nonnegative, and normal wall liquid flux is zero.

Refining 40² to 64² changes the reported maximum surface conductivity increases by less than **0.03 percentage points**. The enlarged-domain 5% case changes its surface potential by about **1 µV** and its conductivity increase by less than **0.01 percentage points**. The code also recovers the independently integrated no-flow fields with decreasing mesh error. These checks concern numerical consistency; the uncertain chemistry and reaction assignment remain experimental questions.

The comparison plots are [114 nA](output/current_partitions/comparison_light_increment.png) and [172 nA](output/current_partitions/comparison_illuminated.png). A full spatial map for the enlarged-domain 5% case is [here](output/current_partitions/illuminated_f0.05_R32_N96_zeta-2VT.png).

## Direct comparison with the Laplace approximation

Run `check_laplace_approximation.py` to compare the saved full-current, 5% ascorbate solution with the constant-conductivity convolution. It does not rerun the ionic transport calculation. The results are in `output/laplace_check/summary.json`, with [surface profiles](output/laplace_check/surface_comparison.png).

The comparison separates three questions:

1. **Prediction from current and bulk conductivity.** Keeping the prescribed 172.158 nA and measured conductivity 435 µS/cm gives a Laplace-convolution centre potential of **17.69 mV**, compared with **12.43 mV** in the transport calculation. The corresponding peak surface electric fields are **140.7 V/m** and **89.8 V/m**. Thus this scenario produces substantial amplitude errors in the constant-conductivity prediction.
2. **Shape after one amplitude adjustment.** Multiplying the entire Laplace potential by the ratio of the two centre potentials gives a maximum surface-potential discrepancy of **5.10% of the computed centre potential**, and a maximum surface-field discrepancy of **12.98% of the computed peak field**, over `0 <= r <= 4a`. This rescaling is a diagnostic of shape; it is not an independently justified conductivity fit. The conductivity that would reproduce the centre value is about 619 µS/cm, compared with the measured 435 µS/cm.
3. **Harmonicity of the bulk potential.** A separate harmonic function is supplied with the computed surface potential and the same remote boundary values as transport. Within `0 <= r,z <= 4a`, it differs from the computed bulk potential by at most **0.337 mV**, or **2.71% of the centre surface potential**. The potential is therefore close to a harmonic extension in this particular absolute-error norm, even though the Ohmic current boundary law gives appreciable amplitude errors. This comparison does not predict the surface potential from the measured current: the computed surface potential is its boundary input.

Here `a = 52.541 µm`, so `4a` is about 210 µm. The potential-shape discrepancy is 5.12%, 5.10%, and 5.10% on the 40², 64², and enlarged-domain 96² calculations; the corresponding field discrepancies are 13.00%, 12.98%, and 12.98%. The bulk harmonic-extension discrepancy is 2.66%, 2.70%, and 2.71%. These stable comparisons distinguish model error from the much smaller observed mesh sensitivity.

The harmonic Stokes construction can still use an auxiliary harmonic extension of the actual surface potential, as described in [HYDRODYNAMICS.md](HYDRODYNAMICS.md). Bulk harmonicity of the physical electric potential is not required for that construction. All comparisons remain conditional on the mixed-reaction hypothesis, dissolved-OPA assumptions, and fixed zeta potential; they do not establish the accuracy of the Laplace approximation in the experiment.
