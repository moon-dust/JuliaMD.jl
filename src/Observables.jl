using BinningAnalysis

mutable struct Observables
    energy::ErrorPropagator{Float64,32}
    magnetization::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    magnetizationVector::LogBinner{Vector{Float64},32,BinningAnalysis.Variance{Vector{Float64}}}
    correlation::LogBinner{Vector{Float64},32,BinningAnalysis.Variance{Vector{Float64}}}
    structureFactor::LogBinner{Array{Float64, 4},32,BinningAnalysis.Variance{Array{Float64, 4}}}
    dynamics::Array{ComplexF64, 5} # [qh qk ql nstep xx(9)]
    chirality::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    spinsAvg::Array{Float64, 2}
end


function Observables(lattice::T) where T<:Lattice
    dim = length(lattice.size)  # ask for dimension
    if dim == 3
        if lattice.calcDyn==true
            return Observables(ErrorPropagator(Float64), LogBinner(Float64), LogBinner(zeros(Float64,3)), LogBinner(zeros(Float64,lattice.length)), 
            LogBinner(zeros(Float64,lattice.calcLim[1]*lattice.size[1], lattice.calcLim[2]*lattice.size[2], lattice.calcLim[3]*lattice.size[3], 3)),
            zeros(ComplexF64, lattice.dynLim[1]*lattice.size[1], lattice.dynLim[2]*lattice.size[2], lattice.dynLim[3]*lattice.size[3], lattice.nstep, 9),
            LogBinner(Float64), zeros(Float64, 3, lattice.length))
        else
            return Observables(ErrorPropagator(Float64), LogBinner(Float64), LogBinner(zeros(Float64,3)), LogBinner(zeros(Float64,lattice.length)), 
            LogBinner(zeros(Float64,lattice.calcLim[1]*lattice.size[1], lattice.calcLim[2]*lattice.size[2], lattice.calcLim[3]*lattice.size[3], 3)),
            im*zeros(1,1,1,1,1), LogBinner(Float64), zeros(Float64, 3, lattice.length))
        end

    elseif dim == 2
        if lattice.calcDyn==true
            return Observables(ErrorPropagator(Float64), LogBinner(Float64), LogBinner(zeros(Float64,3)), LogBinner(zeros(Float64,lattice.length)), 
            LogBinner(zeros(Float64,lattice.calcLim[1]*lattice.size[1], lattice.calcLim[2]*lattice.size[2], 1, 3)),
            zeros(ComplexF64, lattice.dynLim[1]*lattice.size[1], lattice.dynLim[2]*lattice.size[2], 1, lattice.nstep, 9), LogBinner(Float64), zeros(Float64, 3, lattice.length))
        else
            return Observables(ErrorPropagator(Float64), LogBinner(Float64), LogBinner(zeros(Float64,3)), LogBinner(zeros(Float64,lattice.length)), 
            LogBinner(zeros(Float64,lattice.calcLim[1]*lattice.size[1], lattice.calcLim[2]*lattice.size[2], 1, 3)),
            im*zeros(1,1,1,1,1),LogBinner(Float64), zeros(Float64, 3, lattice.length)) 
        end

    elseif dim == 1
        if lattice.calcDyn==true
            return Observables(ErrorPropagator(Float64), LogBinner(Float64), LogBinner(zeros(Float64,3)), LogBinner(zeros(Float64,lattice.length)), 
            LogBinner(zeros(Float64, lattice.calcLim[1]*lattice.size[1], 1, 1, 3)),
            zeros(ComplexF64, lattice.dynLim[1]*lattice.size[1], 1, 1, lattice.nstep, 9),LogBinner(Float64), zeros(Float64, 3, lattice.length))
        else
            return Observables(ErrorPropagator(Float64), LogBinner(Float64), LogBinner(zeros(Float64,3)), LogBinner(zeros(Float64,lattice.length)), 
            LogBinner(zeros(Float64,lattice.calcLim[1]*lattice.size[1], 1, 1, 3)),
            im*zeros(1,1,1,1,1),LogBinner(Float64), zeros(Float64, 3, lattice.length)) 
        end

    end
end

function performMeasurements!(observables::Observables, lattice::T, energy::Float64, sweep::Int) where T<:Lattice
    #measure energy and energy^2
    push!(observables.energy, energy / length(lattice), energy * energy / (length(lattice) * length(lattice)))

    #measure magnetization
    m = getMagnetization(lattice)
    push!(observables.magnetization, norm(m))
    push!(observables.magnetizationVector, m)

    #measure spin correlations, commented out
    # push!(observables.correlation, getCorrelation(lattice))

    #measure fft for correlations
    if sum(lattice.calcLim) != 0
        push!(observables.structureFactor, getFourier(lattice))
    end

    if lattice.chirality == true
        push!(observables.chirality, getChirality(lattice))
    end

    if lattice.spinsAvg == true
        observables.spinsAvg += lattice.spins
    end

    if lattice.calcDyn == true && sweep%100 == 0
        # @show sweep
        observables.dynamics += getDynamics(lattice)
    end
end