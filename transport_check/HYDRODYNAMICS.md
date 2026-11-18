# Hydrodynamics in transport_check

The spatial solver adds incompressible Stokes flow and constant-zeta electroosmotic slip to the same multicomponent, locally equilibrated electrolyte used in the [no-flow calculation](README.md). Individual species, pH, conductivity, diffusion current, and the electric potential remain free to vary.

## Two different questions

The spatial solver supports two operating modes:

1. **Prescribed current:** the input current is held fixed and the electrolyte and flow are solved. This mode does not impose ascorbate depletion. Use `find_depletion_current=False` in `CoupledTransportGrid.solve_coupled()` and supply `initial_current_fraction = I / I_reference`. `I_reference` is a unit for the current argument, not an upper bound. The initial chemical guess in this mode is the uniform reservoir. A positive initial state from a nearby operating point can also be supplied.
2. **First-ascorbate-depletion diagnostic:** the total current is an additional unknown and reconstructed surface ascorbate at the axis is set to zero. Use `find_depletion_current=True`. This finds where the branch connected to the reservoir first exhausts local ascorbate under the imposed reaction closure.

The default sweep in `run_flow_transport.py` performs the second diagnostic. Its approximately 11–15 nA values are **not a hardcoded current ceiling or a whole-cell current prediction**. They are conditional thresholds for the assumption

\[
N_{\mathrm{Asc},z}=-\frac{j_s}{2F},
\]

which assigns all localized current to consumption of fresh ascorbate. Changing that reaction assignment is a physical model change. The solver does not silently replace it with another reaction or accept negative species concentrations.

The prescribed-current mode is separately exercised at 5, 10, and 14 nA in `validate_flow_transport.py`. Those are numerical operating-point checks, not the measured current. Their current stays exactly fixed and surface ascorbate remains positive. Supporting the measured 114–172 nA must be assessed with a consistent reaction assignment, geometry, and transport model.

## Run

First generate the chemical reference from the repository root, then run the spatial flow calculation and its checks:

```powershell
& 'C:/Users/Flori/anaconda3/envs/tube-tracking-clean/python.exe' transport_check/transport_check.py
& 'C:/Users/Flori/anaconda3/envs/tube-tracking-clean/python.exe' transport_check/run_flow_transport.py
& 'C:/Users/Flori/anaconda3/envs/tube-tracking-clean/python.exe' transport_check/validate_flow_transport.py
```

The driver parameter block explicitly lists chemistry/source cases, meshes, domain sizes, zeta potentials, viscosity, permittivity, and nonlinear tolerances. Inputs are inherited from the documented no-flow chemical model. No new conductivity fit is introduced.

## Coupled equations

Species fluxes now contain liquid advection:

\[
\mathbf N_i=-D_i\left(\nabla c_i+z_i c_i\nabla(\phi/V_T)\right)+c_i\mathbf u.
\]

The same conserved component and electrical flux conditions apply at the electrode, including the Tris/TrisH+ reaction. Local acid-base equilibrium and electroneutrality are enforced in every cell. Thus convection contributes to each component flux, while its electrical contribution vanishes because

\[
F\mathbf u\sum_i z_i c_i=0.
\]

The liquid obeys

\[
-\nabla p+\eta\nabla^2\mathbf u=0,
\qquad \nabla\cdot\mathbf u=0,
\]

with the wall conditions

\[
u_z(r,0)=0,
\qquad
u_r(r,0)=-\frac{\varepsilon\zeta}{\eta}E_{s,r}(r),
\qquad E_{s,r}=-\partial_r\phi_s.
\]

The tested constant zeta potentials are approximately −25.7 and −51.4 mV. Zeta is a model input, not inferred from the flow calculation.

## Stokes flow when the bulk potential is not harmonic

The computed physical potential `phi(r,z)` is not assumed harmonic. Instead, introduce a separate auxiliary harmonic function `h` with the same **surface** potential:

\[
\nabla^2h=0,
\qquad h(r,0)=\phi_s(r),
\qquad h\to0\text{ at infinity}.
\]

For the dimensional mobility `mu = -epsilon*zeta/eta`, a Stokes solution is

\[
u_r=-\mu\left(\partial_rh+z\partial_{rz}h\right),
\qquad u_z=-\mu z\partial_{zz}h,
\qquad p-p_\infty=-2\eta\mu\partial_{zz}h.
\]

Equivalently, the axisymmetric streamfunction is

\[
\Psi=\mu z r\partial_rh,
\qquad u_r=-\frac1r\partial_z\Psi,
\qquad u_z=\frac1r\partial_r\Psi.
\]

This follows from a harmonic Papkovich–Neuber representation. Harmonicity of `h` gives the Stokes momentum equation and incompressibility. The factor `z` in the normal velocity gives zero penetration, and the tangential velocity at the wall gives the requested electroosmotic slip. The auxiliary `h` is never substituted for the physical potential in ion migration.

The code scales distances by the measured spot radius `a`, velocity by `D_ref/a`, and potential by `V_T`. Its dimensionless mobility is

\[
\Gamma=-\frac{\varepsilon\zeta V_T}{\eta D_{\mathrm{ref}}}.
\]

Face-integrated liquid fluxes are computed by differences of the same nodal streamfunction. Discrete incompressibility is then an algebraic identity, and normal liquid flux at the electrode is exactly zero.

## Spatial numerical solution

- Axisymmetric finite-volume cells on a mesh refined at the electrode and symmetry axis.
- Eight unknowns per cell: six component totals, pH, and `phi/V_T`.
- Species-level Scharfetter–Gummel diffusion/migration fluxes, projected onto conserved chemical components and electrical current.
- Centered component advection using the conservative Stokes face fluxes. Its electrical contribution is cancelled by electroneutrality. The tested cell Péclet numbers are small.
- An analytical sparse Jacobian for transport, pH, potential, and the optional unknown current.
- Newton iteration with line search preserving nonnegative cell concentrations.
- Alternating Stokes and transport solves until both the liquid fluxes and the freshly evaluated coupled transport residual converge.

The physical surface potential is reconstructed from the first three cell rows. The depletion diagnostic also reconstructs the axis value from radial symmetry. Mesh refinement checks the resulting threshold and slip accuracy.

The finite domain is open. Transport concentrations and potential at remote boundaries are matched to the no-flow far-field solution with the same total current. The Stokes harmonic extension uses its leading harmonic far field. The flow-induced far-field corrections are tested by increasing the outer radius and height from 16 to 32 spot radii. The ceiling and counterelectrode are not explicit walls.

## Results of the ascorbate-only depletion diagnostic

For the unfitted Gaussian case, the finest enlarged domain gives approximately:

| Zeta | Current at first ascorbate depletion | Increase over the same-mesh no-flow value | Maximum wall slip |
| --- | ---: | ---: | ---: |
| 0 | 14.6404 nA | 0 | 0 |
| −25.7 mV | 14.6529 nA | 0.0849% | 0.225 µm/s |
| −51.4 mV | 14.6655 nA | 0.1713% | 0.450 µm/s |

Maximum cell-speed Péclet numbers based on ascorbate diffusivity and spot radius are approximately 0.017 and 0.035. These speeds and Péclet numbers apply **near the ascorbate depletion threshold**, not to a hypothetical solution carrying the full measured current.

The control-calibrated chemical case and the flatter source profile give similarly small relative increases, approximately 0.08–0.18%. Flow therefore changes this particular depletion diagnostic only slightly. It does not justify interpreting the diagnostic as a whole-cell current ceiling.

## Checks

`validate_flow_transport.py` checks the following independently:

1. A manufactured harmonic field with a source below the electrode. Its full half-space Stokes streamfunction and face fluxes are known analytically. Flow errors decrease on 24², 48², and 96² meshes, while normal wall flux stays zero and discrete divergence remains at roundoff.
2. The complete spatial transport/current Jacobian against a mixed chemical, pH, potential, and current directional finite difference. The relative error is approximately 3 × 10^-10.
3. Recovery of the exact no-flow chemical-path solution on successively finer 40², 64², and 96² spatial meshes.
4. Stability of the small flow-induced threshold increase on an enlarged 32-radius, 112²-cell domain. The change in the enhancement fraction is below 3 × 10^-6 in the tested cases.
5. Fresh evaluation of the coupled residual, global component/current conservation, local electroneutrality, positive cell concentrations, and zero normal wall liquid flux. The finest tested coupled residual is about 2 × 10^-11; global component/current imbalance is below 3 × 10^-10 of the imposed reaction/electron rate.
6. Prescribed-current operating points, without the depletion constraint. The supplied current remains fixed.

Checks validate the implementation of this working model. They do not establish the unknown electrode reaction assignment or remove uncertainty in OPA dissolution, mobilities, stock composition, or current localization.

## Files and scope

- `electroosmotic_flow.py`: harmonic surface extension and conservative Stokes streamfunction.
- `coupled_flow_transport.py`: spatial chemistry, component/current fluxes, sparse Jacobian, both current modes, and Stokes/transport iteration.
- `run_flow_transport.py`: parameters and default depletion-diagnostic sweep.
- `validate_flow_transport.py`: independent checks and prescribed-current tests.
- `output/flow/summary.json` and `case_summary.csv`: diagnostic results and convergence histories.
- `output/flow/*.npz`: component and individual-species concentrations, conductivity, physical potential in `states`, velocities, streamfunction, and auxiliary harmonic extension.
- `output/flow/*.png`: spatial concentration, conductivity, potential, and flow maps.
- `output/flow/validation.json`: check results.

This extension includes electroosmotic slip only. It retains the ideal dilute-solution chemistry, local acid-base equilibrium, bulk electroneutrality, constant viscosity/permittivity, and half-space geometry of the working model. It does not resolve the diffuse layer, finite-rate electrode chemistry, diffusioosmotic slip, bulk electrical body forces, the finite chamber ceiling, or product-conversion kinetics.

For the harmonic Stokes representation, see [Gimbutas, Greengard, and Veerapaneni, JFM (2015)](https://arxiv.org/abs/1504.00102). The specific surface-potential construction above is stated explicitly so its equations and wall conditions can be checked directly.
