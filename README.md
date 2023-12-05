# JuliaMD.jl
JuliaMD.jl is a Molecular Dynamics (MD) code to calculate the spectral function for classical spin models. It utilizes the classical Monte Carlo code [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl) to find the ground state, then evolves the spin configuration as a function of time using [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl). Finally, the time evolution in real space is transferred to the frequency domain in reciprocal space through [FFTW.jl](https://juliamath.github.io/FFTW.jl/stable/).

## Installation
To install JuliaMD.jl, type the following command in the Julia REPL:
```julia
] add https://github.com/moon-dust/JuliaMD.jl
```

## Model definition
Like [JuliaSCGA.jl](https://github.com/moon-dust/JuliaMD.jl), the definition of the spin model inherits from [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl). As shown in the example, the *J<sub>1</sub>-J<sub>2</sub>-J<sub>3</sub>* model on a square lattice can be defined as:

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

