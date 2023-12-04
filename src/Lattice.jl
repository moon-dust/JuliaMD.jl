mutable struct DynParm
    calcDyn::Bool   # flag for dynamics calculation
    disorder::Bool  # consider disorder or not
    tau::Float64    # time step for MD calculations
    nstep::Int      # how many time steps
    dynLim::NTuple{3,Int}    # range limit for multiple Brilloin zones
end


mutable struct Lattice{D,N}
    size::NTuple{D,Int} #linear extent of the lattice in number of unit cells
    length::Int #Number of sites N_sites
    unitcell::UnitCell{D}
    sitePositions::Vector{NTuple{D,Float64}}
    parmDyn::DynParm
    calcLim::NTuple{3,Int}
    spinType::String # spin model in "Heisenberg", "Ising", or "softIsing"

    calcDyn::Bool
    disorder::Bool
    tau::Float64
    nstep::Int
    dynLim::NTuple{3,Int}
    more::Float64

    chirality::Bool # calculate chirality or not
    spinsAvg::Bool
    phase::Array{ComplexF64,4} # phase factor for getFourier

    spins::Matrix{Float64} #3*N_sites matrix containing the spin configuration

    interactionSites::Vector{NTuple{N,Int}} #list of length N_sites, for every site contains all interacting sites
    interactionMatrices::Vector{NTuple{N,InteractionMatrix}} #list of length N_sites, for every site contains all interaction matrices
    interactionOnsite::Vector{InteractionMatrix} #list of length N_sites, for every site contains the local onsite interaction matrix
    interactionField::Vector{NTuple{3,Float64}} #list of length N_sites, for every site contains the local field
    Lattice(D,N) = new{D,N}()
end

function Lattice(uc::UnitCell{D}, L::NTuple{D,Int}, calcLim::NTuple{3,Int}, parmDyn::DynParm,  chirality::Bool=false, spinsAvg::Bool=false, spinType::String="Heisenberg", more::Float64=0.0) where D
    #parse interactions
    ##For every basis site b, generate list of sites which b interacts with and store the corresponding interaction sites and matrices. 
    ##Interaction sites are specified by the target site's basis id, b_target, and the offset in units of primitive lattice vectors. 
    ##If b has multiple interactions defined with the same target site, eliminate those duplicates by summing up the interaction matrices. 
    interactionTargetSites = [ Vector{Tuple{Int,NTuple{D,Int},Matrix{Float64}}}(undef,0) for i in 1:length(uc.basis) ] #tuples of (b_target, offset, M)
    for x in uc.interactions
        b1, b2, offset, M = x
        b1 == b2 && offset == Tuple(zeros(D)) && error("Interaction cannot be local. Use setInteractionOnsite!() instead.")
        
        #locate existing coupling to target site and add interaction matrix, applies for two or more M matrices over the same bond
        for i in 1:length(interactionTargetSites[b1])
            if interactionTargetSites[b1][i][1] == b2 && interactionTargetSites[b1][i][2] == offset
                interactionTargetSites[b1][i] = (interactionTargetSites[b1][i][1], interactionTargetSites[b1][i][2], interactionTargetSites[b1][i][3] + M)
                @goto endb1
            end
        end
        #if coupling does not exist yet, push new entry
        push!(interactionTargetSites[b1], (b2, offset, M))
        @label endb1

        #locate existing coupling from target site and add interaction matrix
        for i in 1:length(interactionTargetSites[b2])
            if interactionTargetSites[b2][i][1] == b1 && interactionTargetSites[b2][i][2] == (x->-x).(offset)
                interactionTargetSites[b2][i] = (interactionTargetSites[b2][i][1], interactionTargetSites[b2][i][2], interactionTargetSites[b2][i][3] + transpose(M))
                @goto endb2
            end
        end
        #if coupling does not exist yet, push new entry
        push!(interactionTargetSites[b2], (b1, (x->-x).(offset), transpose(M)))
        @label endb2
    end
    Ninteractions = findmax([ length(interactionTargetSites[i]) for i in 1:length(uc.basis) ])[1]

    #create lattice struct
    lattice = Lattice(D,Ninteractions)
    lattice.size = L
    lattice.length = prod(L) * length(uc.basis)
    lattice.unitcell = uc
    lattice.calcLim = calcLim
    lattice.calcDyn = parmDyn.calcDyn
    lattice.disorder = parmDyn.disorder
    lattice.tau = parmDyn.tau
    lattice.nstep = parmDyn.nstep
    lattice.dynLim = parmDyn.dynLim
    lattice.spinType = spinType
    lattice.chirality = chirality
    lattice.spinsAvg = spinsAvg
    lattice.more = more

    if sum(calcLim) != 0
        nBZ = calcLim

        qh0 = collect(fftshift(fftfreq(L[1])))
        qk0 = collect(fftshift(fftfreq(L[2])))
        ql0 = collect(fftshift(fftfreq(L[3])))

        qh_base = Float64[]
        for ind_h = -ceil(Int16,nBZ[1]/2)+1:floor(Int16,nBZ[1]/2)
            append!(qh_base, qh0.+ind_h)
        end

        qk_base = Float64[]
        for ind_k = -ceil(Int16,nBZ[2]/2)+1:floor(Int16,nBZ[2]/2)
            append!(qk_base, qk0.+ind_k)
        end

        ql_base = Float64[]
        for ind_l = -ceil(Int16,nBZ[3]/2)+1:floor(Int16,nBZ[3]/2)
            append!(ql_base, ql0.+ind_l)
        end

        a1 = [uc.primitive[1]...]
        a2 = [uc.primitive[2]...]
        a3 = [uc.primitive[3]...]
        vol = dot(a1, cross(a2, a3))

        b1 = 2π/vol*cross(a2,a3)
        b2 = 2π/vol*cross(a3,a1)
        b3 = 2π/vol*cross(a1,a2)

        nspin = length(uc.basis)     # spins per cell
        phase = zeros(length(qh_base), length(qk_base), length(ql_base), nspin)*im
        for ind = 1:nspin
            pos = uc.basis[ind]
            
            phase[:,:,:,ind] = [exp(-im*((qh*b1[1]+qk*b2[1]+ql*b3[1])*pos[1] + 
                                         (qh*b1[2]+qk*b2[2]+ql*b3[2])*pos[2] +
                                         (qh*b1[3]+qk*b2[3]+ql*b3[3])*pos[3]))  for qh in qh_base, qk in qk_base, ql in ql_base]
        end
        lattice.phase = phase
    end

  
    #generate linear representation of lattice sites to assign integer site IDs
    ##Enumeration sequence is (a1, a2, ..., b) in row-major fashion
    sites = Vector{NTuple{D+1,Int}}(undef, lattice.length)
    function nextSite(site)
        next = collect(site)
        next[D+1] += 1
        if next[D+1] > length(uc.basis)
            next[D+1] = 1
            next[D] += 1
        end
        for d in reverse(1:D)
            if next[d] >= L[d]
                next[d] = 0
                d-1 > 0 && (next[d-1] += 1)
            end
        end
        return Tuple(next)
    end
    sites[1] = tuple(zeros(Int,D)..., 1)
    for i in 2:length(sites)
        sites[i] = nextSite(sites[i-1])
    end

    #init site positions
    lattice.sitePositions = Vector{NTuple{D,Float64}}(undef, length(sites))
    for i in 1:length(sites)
        site = sites[i]
        lattice.sitePositions[i] = .+([uc.primitive[j] .* site[j] for j in 1:D]...) .+ uc.basis[site[end]]
    end

    #init spins 
    lattice.spins = Array{Float64,2}(undef, 3, length(sites))

    #write interactions to lattice
    lattice.interactionSites = repeat([ NTuple{Ninteractions,Int}(ones(Int,Ninteractions)) ], lattice.length)
    lattice.interactionMatrices = repeat([ NTuple{Ninteractions,InteractionMatrix}(repeat([InteractionMatrix()],Ninteractions)) ], lattice.length)
    lattice.interactionOnsite = repeat([InteractionMatrix()], lattice.length)
    lattice.interactionField = repeat([(0.0,0.0,0.0)], lattice.length)

    function applyPBC(n, L)
        while n < 0; n += L end
        while n >= L; n -= L end
        return n
    end
    function siteIndexFromParametrization(site)
       return findfirst(isequal(site), sites) 
    end

    for i in 1:length(sites)
        site = sites[i]
        b = site[end]

        #onsite interaction
        lattice.interactionOnsite[i] = InteractionMatrix(uc.interactionsOnsite[b])

        #field interaction
        lattice.interactionField[i] = NTuple{3,Float64}(uc.interactionsField[b])

        #two-spin interactions
        interactionSites = repeat([i], Ninteractions)
        interactionMatrices = repeat([InteractionMatrix()], Ninteractions)
        for j in 1:Ninteractions
            if j <= length(interactionTargetSites[b])
                b2, offset, M = interactionTargetSites[b][j]

                primitiveTarget = [applyPBC(site[k] + offset[k], L[k]) for k in 1:D]
                targetSite = tuple(primitiveTarget..., b2) # transfers an array into a tuple

                interactionSites[j] = siteIndexFromParametrization(targetSite) # idx of the TargetSite
                interactionMatrices[j] = InteractionMatrix(M)
            end
        end
        lattice.interactionSites[i] = NTuple{Ninteractions,Int}(interactionSites)
        lattice.interactionMatrices[i] = NTuple{Ninteractions,InteractionMatrix}(interactionMatrices)
    end

    #return lattice
    return lattice
end

function Base.size(lattice::Lattice{D,N}) where {D,N}
    return lattice.size
end

function Base.length(lattice::Lattice{D,N}) where {D,N}
    return lattice.length
end

function calcLim(lattice::Lattice{D,N}) where {D,N}
    return lattice.calcLim
end

function dynLim(lattice::Lattice{D,N}) where {D,N}
    return lattice.dynLim
end

function getSpin(lattice::Lattice{D,N}, site::Int) where {D,N}
    return (lattice.spins[1,site], lattice.spins[2,site], lattice.spins[3,site])
end

function setSpin!(lattice::Lattice{D,N}, site::Int, newState::Tuple{Float64,Float64,Float64}) where {D,N}
    lattice.spins[1,site] = newState[1]
    lattice.spins[2,site] = newState[2]
    lattice.spins[3,site] = newState[3]
end

function getSitePosition(lattice::Lattice{D,N}, site::Int)::NTuple{D,Float64} where {D,N}
    return lattice.sitePositions[site]
end

function getInteractionSites(lattice::Lattice{D,N}, site::Int)::NTuple{N,Int} where {D,N}
    return lattice.interactionSites[site]
end

function getInteractionMatrices(lattice::Lattice{D,N}, site::Int)::NTuple{N,InteractionMatrix} where {D,N}
    return lattice.interactionMatrices[site]
end

function getInteractionOnsite(lattice::Lattice{D,N}, site::Int)::InteractionMatrix where {D,N}
    return lattice.interactionOnsite[site]
end

function getInteractionField(lattice::Lattice{D,N}, site::Int)::NTuple{3,Float64} where {D,N}
    return lattice.interactionField[site]
end