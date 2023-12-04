using JuliaMD, LinearAlgebra, FFTW, HDF5

# serial version


# square cell
a1 = (1.0000, 0.0000) 
a2 = (0.0000, 1.0000) 

uc = UnitCell(a1,a2) 
b1 = addBasisSite!(uc, (0.0000, 0.0000)) 

scaleJ = 5

J1 = [-1.00 -0.00 -0.00; -0.00 -1.00 -0.00; -0.00 -0.00 -1.00]*scaleJ
J2 = [0.50 0.00 0.00; 0.00 0.50 0.00; 0.00 0.00 0.50]*scaleJ
J3 = [0.25 0.00 0.00; 0.00 0.25 0.00; 0.00 0.00 0.25]*scaleJ
J4 = 0.0*[1.00 0.00 0.00; 0.00 1.00 0.00; 0.00 0.00 1.00]*scaleJ

addInteraction!(uc, b1, b1, J1, (1, 0)) 
addInteraction!(uc, b1, b1, J1, (0, 1)) 
addInteraction!(uc, b1, b1, J2, (1, -1)) 
addInteraction!(uc, b1, b1, J2, (1, 1)) 
addInteraction!(uc, b1, b1, J3, (2, 0)) 
addInteraction!(uc, b1, b1, J3, (0, 2)) 
addInteraction!(uc, b1, b1, J4, (1, -2)) 
addInteraction!(uc, b1, b1, J4, (2, -1)) 
addInteraction!(uc, b1, b1, J4, (2, 1)) 
addInteraction!(uc, b1, b1, J4, (1, 2)) 

# superlattice size
L = (40, 40)
calcLim = (0, 0, 0) # (0, 0, 0) for no structure factor calculation
# lattice = Lattice(uc, L, calcLim)

thermalizationSweeps = 50000
measurementSweeps = 1990 # calcDyn per 100 sweeps
# temperature = 2 # Kelvin
# beta = 1/(temperature/11.604)
beta = 2
beta = float(beta) # beta should be float

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


using Interpolations, PyPlot
# read Sαβ
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

omega_fine = range(0, 15, length = nstep-1)

scan_radius = 0.25
theta = range(0, pi, length=50)
scan_pts = cat(scan_radius*cos.(theta), scan_radius*sin.(theta), 0*theta, dims=2)

# distances in lab system
# rl = [sqrt(3) 1; 0 2] # reciprocal lattice in the lab coordinate
rl = [1 0 0; 0 1 0; 0 0 1] # reciprocal lattice in the lab coordinate
#  [ax ay;
#   bx by]
pts_lab = scan_pts*rl
diff_lab = [diff(pts_lab[:,1]) diff(pts_lab[:,2]) diff(pts_lab[:,3])]
dist_lab = accumulate(+, norm.(eachrow(diff_lab)))
pushfirst!(dist_lab, 0)

# norm vector for pts in the lab system
# pts_norm = [norm(pts_lab[i,:]) for i = 1:size(pts_lab,1)]

# interpolate
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

# figure(figsize=(8,3))
figure(figsize=(4,3))
h = pcolor(theta, omega_grid, transpose(int_fine_scale), 
        # cmap = "inferno", 
        cmap = "rainbow", 
        vmin = 0, vmax = 5e4,
        )
xticks([0, pi/4, pi/2, pi*3/4, pi], ["0", "π/4", "2π/4", "3π/4", "π"])

ylabel("E (J)")
cbar = colorbar(h)
cbar.formatter.set_powerlimits((0,0))

# display(gcf())
savefig("square_J123_beta_2_J4_0_curve_serial.png")


