# Transport check

Steady ionic transport around the illuminated ITO region, initially **without liquid flow**. Individual species concentrations, pH, conductivity, diffusion current, and electric potential are calculated together. Conductivity and pH are not held spatially constant.

This is a separate numerical investigation. It does not modify the mathematical SI or the experimental analysis scripts.

The spatial extension with electroosmotic flow, including prescribed-current and first-ascorbate-depletion modes, is documented in [HYDRODYNAMICS.md](HYDRODYNAMICS.md).

The comparison at the measured currents, including an explicitly hypothetical second reaction and an ascorbate-only audit, is documented in [CURRENT_PARTITIONS.md](CURRENT_PARTITIONS.md).

## Run

From the repository root:

```powershell
& 'C:/Users/Flori/anaconda3/envs/tube-tracking-clean/python.exe' transport_check/transport_check.py
& 'C:/Users/Flori/anaconda3/envs/tube-tracking-clean/python.exe' transport_check/validate_transport_check.py
```

The parameter block near the end of `transport_check.py` contains the recipe, diffusivities, pKa values, measurement selection, source profiles, and numerical settings. Dependencies are NumPy, SciPy, and Matplotlib, already available in the project environment. Generated files stay in the ignored `output/` directory.

## What is solved

The electrolyte occupies the half-space `z > 0`. A spatially localized reaction occurs at `z = 0`. The remote electrolyte is a reservoir with the reconstructed buffer composition and electric potential zero. This retains the half-space geometry used in the SI. It is not a finite chamber or counterelectrode calculation.

All concentrations below are **molar concentrations in mol/m³**, numerically equal to mM. Potentials are dimensional volts except for the internal variable `psi = phi / V_T`, where `V_T = RT/F`.

For each dissolved chemical species,

\[
\mathbf N_i=-D_i\left(\nabla c_i+z_i c_i\nabla\psi\right),
\qquad
\mathbf J=F\sum_i z_i\mathbf N_i.
\]

Fast acid-base reactions are assumed to maintain local equilibrium. The model resolves every protonation state of Tris, borate, spermidine, OPA, and ascorbate, along with the oxidation product, H+, and OH-. Their concentrations may vary spatially. The local constraints are

\[
\sum_i z_i c_i=0,
\qquad
\frac{[\text{deprotonated state}]}{[\text{protonated state}]}
=10^{\mathrm{pH}-\mathrm pK_a}.
\]

Electroneutrality is the outer-electrolyte approximation. Nanometre-scale diffuse layers, coating charges, and interfacial capacitors are not resolved in this calculation.

Chemical reactions exchange individual protonation states. Therefore their individual fluxes need not be divergence-free. The **total flux of each conserved chemical component** and the electrical current are divergence-free:

\[
\nabla\cdot\sum_{i\in g}\mathbf N_i=0,
\qquad
\nabla\cdot\mathbf J=0.
\]

The conductivity and diffusion current are evaluated from the calculated species:

\[
\sigma(\mathbf x)=\frac{F}{V_T}\sum_i z_i^2D_i c_i(\mathbf x),
\qquad
\mathbf J_{\mathrm{diff}}=-F\sum_i z_iD_i\nabla c_i.
\]

The electric potential is determined by these transport equations and charge conservation. It is not prescribed by the constant-conductivity Laplace solution.

## Surface reaction, including the TrisH+ source

The baseline reaction is

\[
\mathrm{AscH^-+Tris\longrightarrow DHA+TrisH^++2e^-}.
\]

For a positive reaction rate `q = j_s/(2F)`, the microscopic source vector into the electrolyte is explicitly

| Species | Source per reaction |
| --- | ---: |
| AscH− | −1 |
| Neutral Tris | −1 |
| TrisH+ | +1 |
| Neutral DHA | +1 |

Thus the ascorbate component is consumed, the product component is released, the total Tris component has zero net source, and the electrolyte charge source is `+2Fq`. All other chemical components have zero surface source.

`surface_reaction()` constructs this species source vector before projecting it onto conserved component fluxes and electrical current. The TrisH+ source is therefore included. Under the assumed instantaneous acid-base equilibrium, protons can immediately exchange among Tris and the other buffers. The individual protonation-state fluxes just outside this unresolved equilibration layer need not equal their microscopic reaction-source values. This model imposes the corresponding conserved fluxes. Resolving that layer while imposing each microscopic species flux separately would require finite proton-transfer kinetics.

The boundary conditions outside the equilibrating interfacial region are

\[
N_{\mathrm{Asc},z}=-q,
\qquad N_{\mathrm{product},z}=q,
\qquad N_{g,z}=0\quad\text{for the other components},
\qquad J_z=2Fq.
\]

The `product_anion_limit` sensitivity case replaces the neutral product by a singly negative product and releases two TrisH+ per consumed AscH−. It represents a limiting net reaction with an additional proton released during product conversion. It is not a measured product distribution or a finite-rate DHA hydrolysis calculation.

## Inputs and their status

### Recipe and stock titrations

The recipe comes from the previously supplied local document `C:/Users/Flori/Downloads/KatziProjects_Katzi026 - ShihLab.pdf`, with stock details clarified in the discussion:

- 50% of a nominal 5 mM OPA stock, adjusted to pH 8 with Tris base.
- 0.75 mM final spermidine, supplied as free base.
- 0.5% of a nominal 100 mM ascorbic-acid stock, adjusted to pH 9 with Tris base.
- 1.5% of a 1 M Tris stock, adjusted to pH 9.3 with boric acid.
- Tween is treated as neutral. Its micelles, adsorption, and effects on mobilities are not modelled.

The reconstructed Tris additions are based on ideal charge balance at the stated stock pH values. This implements the previously accepted hypothesis that the reported 450 mM Tris label on the ascorbate stock is erroneous. It does not treat that hypothesis as experimentally verified. The model requires approximately 974 mM Tris in that nominal 100 mM ascorbate stock at 25 C.

The final baseline totals are approximately 26.89 mM Tris, 1.522 mM boron, 0.75 mM spermidine, 2.5 mM dissolved OPA, and 0.5 mM ascorbate. The model reservoir pH is approximately 8.815. This pH is a reconstruction, not a measurement of the final buffer.

All nominal OPA is assumed freely dissolved and available for acid-base equilibrium. This is uncertain. OPA aggregation, limited dissolution, Tris–borate complexes, activity coefficients, and ion pairing are omitted. Fixed chemical charge in the coating is distinct from this assumed dissolved OPA.

### Electrochemical data and source geometry

- Spot half-maximum radius: **52.541 µm**, from the previously background-corrected illumination image and the user-supplied calibration of 221 µm per 2048 pixels.
- Current CSVs: `ito_light_response/data/ito_characterization_1/voltage_sweep/on900s.csv` and `off900s.csv`.
- Selected measurements satisfy **|V − 2.0 V| < 0.05 V**. CH2 is converted with the existing 1 kOhm shunt and 1000x amplifier gain.
- Resulting means are approximately 172.2 nA illuminated and 58.3 nA dark, with an illumination-induced increase of 113.9 nA.
- Measured conductivity: **435 µS/cm** with ascorbate stock and **390 µS/cm** without it.

The 2 V electrode bias selects the measured current. It is not imposed as the outer-electrolyte surface potential. The interfacial kinetic law is not known well enough to calculate the current from that bias here. Instead, the calculation tests which prescribed ascorbate currents can be sustained by transport.

An illumination image is not a spatial current measurement. Gaussian and flatter positive current profiles are tested with the same measured half-maximum radius. The flatter profile has the form

\[
p_m(r)=\frac{1-\left(1-e^{-r^2/b_m^2}\right)^m}{\pi b_m^2 H_m},
\qquad
H_m=\sum_{k=1}^{m}\frac1k,
\]

where `b_m` is chosen to make the half-maximum radius equal to the measured radius. The cases are `m = 1` and `m = 16`. Each profile integrates to one, and `j_s = I p_m`.

Uniform-flux and transport-limited absorbing disks of the same radius are also reported as geometric references. The absorbing disk has edge-enhanced flux, not uniform current density.

### Chemical parameters and conductivity calibration

| Family | pKa values | Diffusivity (10⁻⁹ m²/s) | Status |
| --- | --- | ---: | --- |
| Tris | 8.06 | 0.758 | Representative mobility estimate; same value used for both forms |
| Boric acid/borate | 9.237 | 1.0 | Representative assumption |
| Spermidine | 8.25, 9.71, 10.90 | 0.49 | Representative literature parameters |
| Dissolved OPA | 2.6, 8.0 | 0.50 | Uncertain model assumptions |
| Ascorbate | 4.17, 11.57 | 0.66 | Representative literature parameters |
| Neutral product | none | 0.66 | Assumed |
| H+ / OH− | water pKw = 14 | 9.31 / 5.27 | Standard dilute-water estimates |

The baseline does **not** fit conductivity. Its calculated reservoir conductivity is approximately **377.7 µS/cm**, compared with the measured 435 µS/cm.

Only the explicitly named `control_calibrated` case adjusts the effective OPA diffusivity to reproduce the **390 µS/cm control without ascorbate stock**. This gives approximately `0.623 × 10^-9 m²/s`. The same value then predicts approximately **419.2 µS/cm** after adding the ascorbate stock. The fitted parameter can absorb other model errors and is not an independently measured OPA mobility. Neither the current limit nor the local conductivity change is fitted.

Other cases vary OPA pKa2 between 7.5 and 8.5, its assumed diffusivity between 0.35 and 0.80 × 10^-9 m²/s, and the neutral versus singly charged product limit.

## Exact spatial reduction in the absence of flow

The no-flow problem has a useful structure. All conserved surface fluxes have the same spatial profile, with ratios fixed by reaction stoichiometry. We construct a harmonic transport coordinate `s(x)` satisfying

\[
\nabla^2s=0,
\qquad -D_*\partial_zs=\frac{j_s}{nF},
\qquad s\to0\ \text{at infinity}.
\]

Here `s` has units mol/m³, `D_*` is a reference diffusivity, and `n = 2`. **This auxiliary coordinate is harmonic. The physical electric potential is not assumed harmonic.**

We seek every local concentration and `psi` as functions of `s`. Substitution into the full Nernst–Planck flux gives

\[
\mathbf N_i=-D_i\left(\frac{dc_i}{ds}+z_i c_i\frac{d\psi}{ds}\right)\nabla s.
\]

For each component, we set the coefficient multiplying `−D_* grad(s)` equal to its net production coefficient in the surface reaction. The corresponding charge coefficient is `n`. With `d_i = D_i/D_*`, the equations along the chemical path are

\[
\sum_{i\in g}d_i\left(\frac{dc_i}{ds}+z_i c_i\frac{d\psi}{ds}\right)=\nu_g,
\]

\[
\sum_i z_i d_i\left(\frac{dc_i}{ds}+z_i c_i\frac{d\psi}{ds}\right)=n,
\qquad \frac{d}{ds}\sum_i z_i c_i=0.
\]

The component coefficients are `nu_Asc = -1`, `nu_product = +1`, and zero for the other components. Local acid-base equilibrium supplies the individual state concentrations as functions of component totals and pH. At each path point, the code solves an 8 × 8 linear system for derivatives of the six component totals, pH, and `psi`, and integrates that nonlinear path numerically from the reservoir.

This constructs a solution of the spatial equations: all component fluxes and electrical current are constant multiples of `grad(s)`, so their divergences vanish. Their normal surface components equal the imposed reaction fluxes. The local charge and equilibrium constraints hold along the path. This is a reduction of the stated no-flow boundary-value problem, not a planar or radial approximation to the source geometry.

The source coordinate is mapped into the axisymmetric half-space by a Hankel integral of the specified current profile. The chemical path is integrated only to the first zero of total ascorbate. Currents requiring a larger coordinate have no nonnegative solution on this branch and are reported as unsupported. Negative extrapolated concentrations are not accepted as solutions.

Flow generally breaks this reduction. The coupled spatial solver is documented in [HYDRODYNAMICS.md](HYDRODYNAMICS.md). Finite homogeneous reaction rates or nonproportional surface reactions can also break the no-flow reduction.

## Results from the current defaults

| Case | Reservoir conductivity (µS/cm) | Conductivity change at first ascorbate depletion | Absorbing-disk current limit (nA) |
| --- | ---: | ---: | ---: |
| Unfitted baseline | 377.7 | +0.56% | 13.756 |
| Control-conductivity calibration | 419.2 | +1.48% | 13.754 |
| OPA pKa2 = 7.5 | 395.9 | +1.13% | 13.735 |
| OPA pKa2 = 8.5 | 340.8 | −0.67% | 13.806 |
| OPA diffusivity = 0.35 × 10^-9 m²/s | 326.9 | −0.90% | 13.758 |
| OPA diffusivity = 0.80 × 10^-9 m²/s | 479.1 | +2.51% | 13.751 |
| Singly negative product limit | 377.7 | +6.11% | 13.726 |

For the unfitted baseline, first depletion occurs at approximately **14.64 nA for the Gaussian profile**, **11.36 nA for the flatter profile**, **10.80 nA for uniform disk current density**, and **13.76 nA for the absorbing-disk reference**. These thresholds include migration and the coupled buffer/ion response. They remain far below both the 113.9 nA illumination-induced increase and the 172.2 nA illuminated whole-cell current if those currents are assigned entirely to ascorbate oxidation at this spot.

At first depletion in the baseline chemical path, the local pH changes from approximately 8.815 to 8.793. TrisH+ increases from approximately 4.017 to 4.158 mM, while dissolved OPA accumulates and spermidine is depleted. Their conductivity contributions partly cancel the loss of ascorbate. The net conductivity change is approximately +2.10 µS/cm, or +0.56%, despite complete ascorbate depletion at that point.

These results distinguish two statements: individual concentrations can vary substantially while conductivity varies little. They do not show that the no-flow model can explain the measured current.

Diffusion still supplies approximately **18% of the electrical current in the unfitted baseline**, or **9% in the control-calibrated case**, along the physical path. Thus small conductivity variation alone does not justify dropping diffusion current. The corresponding surface potential is about 1.4 mV at the supported depletion threshold. It is **not** a recalculated surface potential for the measured 114–172 nA current, for which this ascorbate-only no-flow model has no nonnegative steady solution.

The earlier passive-ascorbate calculation in `tmp/` held the field generated by the larger measured current fixed and asked how much ascorbate that field could supply. This calculation instead requires ascorbate oxidation to carry the entire prescribed current and solves the field with the other ions. Their numerical limits answer different boundary-value problems.

## Validation

`validate_transport_check.py` checks:

1. Acid-fraction derivatives against finite differences.
2. The explicit Tris consumption and TrisH+ source, component stoichiometry, and two-electron charge balance.
3. An independent exact neutral-diffusion problem, including its depletion threshold and product concentration.
4. Chemical-path integration at three tolerances.
5. Local component/current flux balances and electroneutrality throughout the path.
6. The original spatial component and current equations on separate 24², 48², 96², and 192² axisymmetric finite-volume meshes. These use species-level Scharfetter–Gummel fluxes, not the chemical-path matrix.

The largest integrated finite-volume residual, normalized by the imposed component or electron rate, decreases from approximately 0.00765 to 0.00225, 0.000610, and 0.000159 under mesh refinement. This is consistent with second-order convergence. Local path balance errors are approximately 10^-15 and the baseline charge residual is below 10^-12 mol/m³.

The validation checks numerical implementation of the stated model. It does not establish the uncertain dissolved-OPA chemistry, stock composition, current localization, reaction assignment, or product kinetics experimentally.

## Outputs

- `output/summary.json`: full inputs, stock reconstruction, explicit surface reactions, reservoir and depletion compositions, current thresholds, measured-current feasibility, and diagnostics for every case.
- `output/case_summary.csv`: compact numerical comparison.
- `output/unfitted_fields.npz` and `control_calibrated_fields.npz`: axisymmetric fields, including every individual species, component totals, pH, electric potential, and conductivity. Species and state labels are stored in the archives. The plotted current is 95% of each Gaussian-profile depletion threshold.
- `output/*_fields.png`: spatial maps for the unfitted and control-calibrated cases.
- `output/chemical_response.png`: conductivity, pH, and potential along the chemical path.
- `output/conductivity_breakdown.png`: changes in the conductivity contributions of the ion families.
- `output/validation.json`: independent numerical validation results.

## References and limitations

- [Conceição et al., Analytical Chemistry (2024), DOI 10.1021/acs.analchem.4c01645](https://pmc.ncbi.nlm.nih.gov/articles/PMC11411494/): measured ascorbate diffusivity near 6.6 × 10^-10 m²/s and two-electron oxidation under its experimental conditions. These values are representative inputs, not measurements in this OPA/Tween buffer.
- [Reference values of the diffusion-limited current at a microdisk electrode](https://www.sciencedirect.com/science/article/pii/S0013468604008308): the neutral diffusion-only disk benchmark `I = 4nFDca`.
- [Labadi et al., Acta Chemica Scandinavica 45, 1055–1059 (1991)](https://actachemscand.org/pdf/acta_vol_45_p1055-1059.pdf): representative spermidine protonation constants and their dependence on measurement conditions, including literature values tabulated by these authors.
- [COMSOL, Domain equations for Nernst–Planck transport and electroneutrality](https://doc.comsol.com/6.3/doc/com.comsol.help.echem/echem_ug_electrochem.06.088.html): standard transport and charge-conservation formulation. The implementation here is independent Python code.
- [Cioffi et al., Analyst (2000), DOI 10.1039/B007299K](https://pubs.rsc.org/en/content/articlelanding/2000/an/b007299k) and [Dewhirst and Fry, Biochemical Journal (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6225978/): DHA conversion to charged degradation products. The neutral-product baseline and the instantaneous singly charged product case are limiting models; finite-rate degradation at the actual pH is unresolved.

The present version omits liquid flow, the finite chamber and counterelectrode geometry, double-layer structure, finite acid-base kinetics, product hydrolysis kinetics, alternative electrode reactions, surface adsorption, and nonideal solution chemistry. It is a coupled bulk-transport calculation for the specified working model, not a complete microscopic model of the electrode.
