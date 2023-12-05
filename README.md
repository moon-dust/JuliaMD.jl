# JuliaMD.jl
JuliaMD.jl is a Molecular Dynamics (MD) code to calculate the spectral function for classical spin models. It utilizes the classical Monte Carlo code [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl) to find the ground state, then evolves the spin configuration as a function of time using [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl). Finally, the time evolution in real space is transferred to the frequency domain in reciprocal space through [FFTW.jl](https://juliamath.github.io/FFTW.jl/stable/).

## Installation
To install JuliaMD.jl, type the following command in the Julia REPL:
```julia
] add https://github.com/moon-dust/JuliaMD.jl
```

## Model definition
JuliaMD follows the script structure, including the definition of the spin model, of [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl). As shown in the example script, the *J<sub>1</sub>-J<sub>2</sub>-J<sub>3</sub>* model on a square lattice can be defined as:

```julia
# square cell
a1 = (1.0000, 0.0000) 
a2 = (0.0000, 1.0000) 

uc = UnitCell(a1,a2) 
b1 = addBasisSite!(uc, (0.0000, 0.0000)) 

scaleJ = 5
J1 = [-1.00 0.00 0.00; 0.00 -1.00 0.00; 0.00 0.00 -1.00]*scaleJ
J2 = [0.50 0.00 0.00; 0.00 0.50 0.00; 0.00 0.00 0.50]*scaleJ
J3 = [0.25 0.00 0.00; 0.00 0.25 0.00; 0.00 0.00 0.25]*scaleJ

addInteraction!(uc, b1, b1, J1, (1, 0)) 
addInteraction!(uc, b1, b1, J1, (0, 1)) 
addInteraction!(uc, b1, b1, J2, (1,-1)) 
addInteraction!(uc, b1, b1, J2, (1, 1)) 
addInteraction!(uc, b1, b1, J3, (2, 0)) 
addInteraction!(uc, b1, b1, J3, (0, 2)) 
```

## Monte Carlo simulations including the MD calculations
Compared to [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl), additional parameters are introduced to the *Lattice* function for convenience, including *calcLim*, *parmDyn*, *chirality*, *spinsAvg*, and *spinType*. Among them, *parmDyn* is directly related to the MD calculations. This is an object parameter composed of *calcDyn (boolean)*, *disorder (boolean)*, *tau* (step length for time evolution), *nstep* (number of steps), *dynLim* (limit for the MD calculations).

```julia
# superlattice size
L = (40, 40)
calcLim = (0, 0, 0) # (0, 0, 0) for no structure factor calculation

thermalizationSweeps = 50000
measurementSweeps = 1990 # calcDyn per 100 sweeps

# define the temperature parameter
beta = 2.0

spinType="Heisenberg"

calcDyn = true
disorder = false
tau=0.1
nstep= 401
dynLim = (1, 1, 1)
parmDyn = DynParm(calcDyn, disorder, tau, nstep, dynLim)

chirality = false
spinsAvg = false
lattice = Lattice(uc, L, calcLim, parmDyn, chirality, spinsAvg, spinType)
m = MonteCarlo(lattice, beta, thermalizationSweeps, measurementSweeps)
run!(m)
```

## Define interpolation functions on the FFTW grid

```julia
using Interpolations, PyPlot
# read Sαβ from Monte Carlo
Sαβ = m.observables.dynamics
n_run = measurementSweeps

# qh, qk, and omega basis for interpolation
dBZ = lattice.dynLim
L = lattice.size
qh0 = collect(fftshift(fftfreq(L[1])))
qk0 = collect(fftshift(fftfreq(L[2])))

qh_base = copy(qh0)
if dBZ[1] > 1
    for ind_h = 1:dBZ[1]-1
        append!(qh_base, qh0.+ind_h)
    end
end

qk_base = copy(qk0)
if dBZ[2] > 1
    for ind_k = 1:dBZ[2]-1
        append!(qk_base, qk0.+ind_k)
    end
end

n_timestep = size(Sαβ,4)
timestep = 0.1*1^-1
omega_base = fftshift(fftfreq(n_timestep)).*2π./timestep

# which L plane to plot
idL = 1

Sxx_interp = LinearInterpolation((qh_base, qk_base, omega_base), real(Sαβ[:,:,idL,:,1])/n_run, extrapolation_bc = Line())
Sxy_interp = LinearInterpolation((qh_base, qk_base, omega_base), real(Sαβ[:,:,idL,:,2])/n_run, extrapolation_bc = Line())
Sxz_interp = LinearInterpolation((qh_base, qk_base, omega_base), real(Sαβ[:,:,idL,:,3])/n_run, extrapolation_bc = Line())

Syx_interp = LinearInterpolation((qh_base, qk_base, omega_base), real(Sαβ[:,:,idL,:,4])/n_run, extrapolation_bc = Line())
Syy_interp = LinearInterpolation((qh_base, qk_base, omega_base), real(Sαβ[:,:,idL,:,5])/n_run, extrapolation_bc = Line())
Syz_interp = LinearInterpolation((qh_base, qk_base, omega_base), real(Sαβ[:,:,idL,:,6])/n_run, extrapolation_bc = Line())

Szx_interp = LinearInterpolation((qh_base, qk_base, omega_base), real(Sαβ[:,:,idL,:,7])/n_run, extrapolation_bc = Line())
Szy_interp = LinearInterpolation((qh_base, qk_base, omega_base), real(Sαβ[:,:,idL,:,8])/n_run, extrapolation_bc = Line())
Szz_interp = LinearInterpolation((qh_base, qk_base, omega_base), real(Sαβ[:,:,idL,:,9])/n_run, extrapolation_bc = Line())

```
## Define a scanning line for the spectra plot

```julia

# define a circular line along the spiral surface
scan_radius = 0.25
theta = range(0, pi, length=50)
scan_pts = cat(scan_radius*cos.(theta), scan_radius*sin.(theta), 0*theta, dims=2)

# transfer to the lab system
rl = [1 0 0; 0 1 0; 0 0 1] # reciprocal lattice in the lab coordinate
pts_lab = scan_pts*rl
diff_lab = [diff(pts_lab[:,1]) diff(pts_lab[:,2]) diff(pts_lab[:,3])]
dist_lab = accumulate(+, norm.(eachrow(diff_lab)))
pushfirst!(dist_lab, 0)

# interpolate along the scanning line
omega_fine = range(0, 15, length = nstep-1)
int_fine = zeros(size(scan_pts,1), size(omega_fine,1))
for idx_pts = 1:size(scan_pts,1), idx_omega = 1:size(omega_fine,1)
    pts_norm = norm(pts_lab[idx_pts,:])
    if pts_norm != 0
        pts_nvec = pts_lab[idx_pts,:]/pts_norm
    else
        pts_nvec = pts_lab[idx_pts,:]
    end

    # find the corresponding pts in the first BZ
    pts_1BZ = scan_pts[idx_pts,:] 

    int_fine[idx_pts, idx_omega] = (Sxx_interp(pts_1BZ[1], pts_1BZ[2], omega_fine[idx_omega])*(1-pts_nvec[1]*pts_nvec[1])
                                + 2*Sxy_interp(pts_1BZ[1], pts_1BZ[2], omega_fine[idx_omega])*(0-pts_nvec[1]*pts_nvec[2])
                                +   Syy_interp(pts_1BZ[1], pts_1BZ[2], omega_fine[idx_omega])*(1-pts_nvec[2]*pts_nvec[2])
                                +   Szz_interp(pts_1BZ[1], pts_1BZ[2], omega_fine[idx_omega]) )
end
int_fine_global = int_fine

dist_grid = transpose(repeat(dist_lab, 1, length(omega_fine)))
omega_grid = repeat(collect(omega_fine), 1, length(dist_lab))

# scale by the omega factor
int_fine_scale = int_fine_global.*reshape(omega_fine,1,length(omega_fine))

```

## Plot

```julia

figure(figsize=(4,3))
h = pcolor(theta, omega_grid, transpose(int_fine_scale), 
        cmap = "rainbow", 
        vmin = 0, vmax = 5e4,
        )
xticks([0, pi/4, pi/2, pi*3/4, pi], ["0", "π/4", "2π/4", "3π/4", "π"])

ylabel("E (J)")
cbar = colorbar(h)
cbar.formatter.set_powerlimits((0,0))

display(gcf())
```
