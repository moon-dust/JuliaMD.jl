module JuliaMD

include("UnitCell.jl")
export UnitCell, addInteraction!, setInteractionOnsite!, setField!, addBasisSite!
include("InteractionMatrix.jl")
export coefMatrix
include("Lattice.jl")
export Lattice, DynParm, size, length, getSpin, setSpin!, getSitePosition

include("Observables.jl")
export Observables
include("Spin.jl")
export getEnergy, getMagnetization, getCorrelation, getFourier

include("MonteCarlo.jl")
export MonteCarlo, run!

include("Helper.jl")
include("IO.jl")
export writeMonteCarlo, readMonteCarlo

using Reexport
@reexport using BinningAnalysis

end