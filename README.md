# JuliaMD.jl
JuliaMD.jl is a Molecular Dynamics (MD) code to calculate the spectral function for classical spin models. It utilizes the classical Monte Carlo code [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl) to find the ground state, then evolves the spin configuration as a function of time using [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl). Finally, the time evolution in real space is transferred to the frequency domain in reciprocal space through [FFTW.jl](https://juliamath.github.io/FFTW.jl/stable/).

## Installation
To install JuliaMD.jl, type the following command in the Julia REPL:
```julia
] add https://github.com/moon-dust/JuliaMD.jl
```

## Model definition
Like that in [JuliaSCGA.jl](https://github.com/moon-dust/JuliaMD.jl), the definition of the spin model inherits from [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl). As shown in the example, the *J<sub>1</sub>-J<sub>2</sub>-J<sub>3</sub>* model on a square lattice can be defined as:

```julia
# square cell
a1 = (1.0000, 0.0000) 
a2 = (0.0000, 1.0000) 

uc = UnitCell(a1,a2) 
b1 = addBasisSite!(uc, (0.0000, 0.0000)) 

scaleJ = 5
J1 = [-1.00 -0.00 -0.00; -0.00 -1.00 -0.00; -0.00 -0.00 -1.00]*scaleJ
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
Classical Monte Carlo simulations follow the steps in [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl). Additional parameters are introduced to the *Lattice* function for convenience, including *calcLim*, *parmDyn*, *chirality*, *spinsAvg*, and *spinType*. Among them, *parmDyn* is directly related to the MD calculations. This is an object parameter composed of *calcDyn (boolean)*, *disorder (boolean)*, *\tau* (step length for time evolution), *nstep* (number of steps), *dynLim* (limit for the MD calculations)

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
