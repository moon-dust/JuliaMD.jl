using Random
using LinearAlgebra
using FFTW
using DifferentialEquations

function uniformOnSphere(rng = Random.GLOBAL_RNG)
    phi = 2.0 * pi * rand(rng) # rand(rng) generates a random number between 0 and 1
    z = 2.0 * rand(rng) - 1.0;
    r = sqrt(1.0 - z * z)
    return (r * cos(phi), r * sin(phi), z)
end

function randomIsing(rng = Random.GLOBAL_RNG)
    return (0.0, 0.0, rand(rng,(1.0,-1.0)))
end

function softIsing(rng = Random.GLOBAL_RNG)
    return (0.0, 0.0, 2.0*rand(rng) - 1.0)
end

function mirrorCone(rng = Random.GLOBAL_RNG)
    phi = 2.0 * pi * rand(rng) # rand(rng) generates a random number between 0 and 1
    y = sqrt(3)/2*sign(rand(rng)-0.5)
    r = 1/2
    return (r * cos(phi), y, r * sin(phi))
end

function exchangeEnergy(s1, M::InteractionMatrix, s2)::Float64
    return s1[1] * (M.m11 * s2[1] + M.m12 * s2[2] + M.m13 * s2[3]) + s1[2] * (M.m21 * s2[1] + M.m22 * s2[2] + M.m23 * s2[3]) + s1[3] * (M.m31 * s2[1] + M.m32 * s2[2] + M.m33 * s2[3])
end

function getEnergy(lattice::Lattice{D,N})::Float64 where {D,N}
    energy = 0.0

    for site in 1:length(lattice)
        s0 = getSpin(lattice, site)

        #two-spin interactions
        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        for i in 1:length(interactionSites)
            if site > interactionSites[i] # avoid double counting
                energy += exchangeEnergy(s0, interactionMatrices[i], getSpin(lattice, interactionSites[i]))
            end
        end

        #onsite interaction
        energy += exchangeEnergy(s0, getInteractionOnsite(lattice, site), s0)

        #field interaction
        energy += dot(s0, getInteractionField(lattice, site))

        #artificial single-ion anisotropy
        if lattice.more != 0
            theta = acos(s0[3])
            phi = acos(s0[1])
            energy += lattice.more*sin(theta)^2*cos(phi)^2
        
            # confine spins to the constant |Sy| cone
            # energy += lattice.more*(abs(s0[2])-sqrt(3)/2)^2
        end
            
    end

    return energy
end

function getEnergyDifference(lattice::Lattice{D,N}, site::Int, newState::Tuple{Float64,Float64,Float64})::Float64 where {D,N}
    dE = 0.0
    oldState = getSpin(lattice, site)
    ds = newState .- oldState

    #two-spin interactions
    interactionSites = getInteractionSites(lattice, site)
    interactionMatrices = getInteractionMatrices(lattice, site)
    for i in 1:length(interactionSites)
        dE += exchangeEnergy(ds, interactionMatrices[i], getSpin(lattice, interactionSites[i]))
    end

    #onsite interaction
    interactionOnsite = getInteractionOnsite(lattice, site)
    dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    #field interaction
    dE += dot(ds, getInteractionField(lattice, site))

    # artificial single-ion anisotropy
    if lattice.more != 0
        theta_old = acos(oldState[3])
        phi_old = acos(oldState[1])
        theta_new = acos(newState[3])
        phi_new = acos(newState[1])
        dE += lattice.more*(sin(theta_new)^2*cos(phi_new)^2 - sin(theta_old)^2*cos(phi_old)^2)

        # confine spins to the constant |Sy| cone
        # sy_old = oldState[2]
        # sy_new = newState[2]
        # dE += lattice.more*((abs(sy_new)-sqrt(3)/2)^2 - (abs(sy_old)-sqrt(3)/2)^2)
    end    

    return dE
end

function getMagnetization(lattice::Lattice{D,N}) where {D,N}
    mx, my, mz = 0.0, 0.0, 0.0
    for i in 1:length(lattice)
        spin = getSpin(lattice, i)
        mx += spin[1]
        my += spin[2]
        mz += spin[3]
    end
    return [mx, my, mz] / length(lattice)
end


function getCorrelation(lattice::Lattice{D,N}, spin::Int = 1) where {D,N}
    corr = zeros(length(lattice))
    s0 = getSpin(lattice, spin)
    for i in 1:length(lattice)
        corr[i] = dot(s0, getSpin(lattice, i))
    end
    return corr
end

function getChirality(lattice::Lattice{D,N}) where {D,N}

    # ~~~~~~~~~~~~~~~~~~~~~  chirality for the smaller triangles in Gd3Ru4Al12 ~~~~~~~~~~~~~~~~~~~
    #=
    nCell = size(lattice)[1]*size(lattice)[2]*size(lattice)[3]

    chirality::Float64 = 0.0
    for i in 1:nCell*Int(2)
        spin1 = getSpin(lattice, 3*i-2)
        spin2 = getSpin(lattice, 3*i-1)
        spin3 = getSpin(lattice, 3*i)

        chirality += (spin1[1]*(spin2[2]*spin3[3]-spin2[3]*spin3[2]) + 
                      spin1[2]*(spin2[3]*spin3[1]-spin2[1]*spin3[3]) +
                      spin1[3]*(spin2[1]*spin3[2]-spin2[2]*spin3[1]))
    end
    return chirality/nCell/Int(2)
    =#

    # ~~~~~~~~~~~~~~~~~~~ chirality for the Cs3Fe2Cl9 lattice ~~~~~~~~~~~~~~~~~~~~~~~~
    
    L = lattice.size
    nAtom = length(lattice.unitcell.basis)
    spins_layer = lattice.spins[:,1:L[3]*nAtom:end]
    # site_layer = lattice.sitePositions[1:L[3]*nAtom:end]
    # site_layer = hcat(collect.(site_layer)...)

    # take out only one sublattice
    # if mod(L[1],3) != 0 || mod(L[2],3) != 0
    #     error("chirality calculation requires a 3N-by-3N lattice")
    # end
    # idx_sub = Int.(zeros(Int(L[1]*L[2]/3)));
    # for idx = 1:Int(L[2]/3)
    #     col = [1:3:L[1]; L[1]+3:3:2*L[1]; 2*L[1]+2:3:3*L[1]] .+ 3*L[2]*(idx-1)
    #     idx_sub[1+(idx-1)*L[2]:L[2]*idx] = Int.(col)
    # end

    # site_sub = site_layer[:,idx_sub]
    # spins_sub = spins_layer[:,idx_sub]

    # chirality = 0.0;
    # for idx_x = 1:Int(L[2]/3)-1
    #     for idx_y = 1:Int(L[1]/3)-1
    #         ind1 = (idx_x - 1)*L[1] + idx_y
    #         ind2 = (idx_x)*L[1] + (idx_y + 1)
    #         ind3 = (idx_x - 1)*L[1] + (idx_y + 1)
    #         ind4 = (idx_x)*L[1] + idx_y
    #         chirality += dot(spins_sub[:, ind1], cross(spins_sub[:,ind2], spins_sub[:,ind3])) + 
    #                      dot(spins_sub[:, ind1], cross(spins_sub[:,ind4], spins_sub[:,ind2]))
    #     end
    # end

    chirality = 0.0
    for idx_x = 1:L[2]-1
        for idx_y = 1:L[1]-1
            ind1 = (idx_x - 1)*L[1] + idx_y
            ind2 = (idx_x)*L[1] + (idx_y + 1)
            ind3 = (idx_x - 1)*L[1] + (idx_y + 1)
            ind4 = (idx_x)*L[1] + idx_y
            chirality += dot(spins_layer[:, ind1], cross(spins_layer[:,ind2], spins_layer[:,ind3])) + 
                         dot(spins_layer[:, ind1], cross(spins_layer[:,ind4], spins_layer[:,ind2]))
        end
    end

    return abs(chirality/(L[2]-1)/(L[1]-1))
    

    # ~~~~~~~~~~~~~~  chirality for the triangular lattice ~~~~~~~~~~~~~
    #=
    L = lattice.size
    nAtom = length(lattice.unitcell.basis)
    spins_layer = lattice.spins

    # take out only one sublattice
    if mod(L[1],3) != 0 || mod(L[2],3) != 0
        error("chirality calculation requires a 3N-by-3N lattice")
    end
    # idx_sub = Int.(zeros(Int(L[1]*L[2]/3)));
    # for idx = 1:Int(L[2]/3)
    #     col = [1:3:L[1]; L[1]+3:3:2*L[1]; 2*L[1]+2:3:3*L[1]] .+ 3*L[2]*(idx-1)
    #     idx_sub[1+(idx-1)*L[2]:L[2]*idx] = Int.(col)
    # end
    # spins_sub = spins_layer[:,idx_sub]

    chirality = 0.0;
    for idx_x = 1:L[2]-1
        for idx_y = 1:L[1]-1
            ind1 = (idx_x - 1)*L[1] + idx_y
            ind2 = (idx_x)*L[1] + (idx_y + 1)
            ind3 = (idx_x - 1)*L[1] + (idx_y + 1)
            ind4 = (idx_x)*L[1] + idx_y
            chirality += dot(spins_layer[:, ind1], cross(spins_layer[:,ind2], spins_layer[:,ind3])) + 
                         dot(spins_layer[:, ind1], cross(spins_layer[:,ind4], spins_layer[:,ind2]))
        end
    end
    return abs(chirality/(L[2]-1)/(L[1]-1))
    =#

end

function getFourier(lattice::Lattice{D,N}) where {D,N}
    
    nspin = length(lattice.unitcell.basis)     # spins per cell
    # ncell = lattice.size[1]*lattice.size[2]     # number of cells
    nBZ = lattice.calcLim

    if D==1  # 1D chain
        # initialize
        str_fac_x = zeros(nBZ[1]*lattice.size[1])*im
        str_fac_y = zeros(nBZ[1]*lattice.size[1])*im
        str_fac_z = zeros(nBZ[1]*lattice.size[1])*im
        sf2 = zeros(nBZ[1]*lattice.size[1])

        a1 = lattice.unitcell.primitive[1]
        b1 = 2π/a1[1]

        qh0 = collect(fftshift(fftfreq(lattice.size[1])))

        qh_base = copy(qh0)
        if nBZ[1] > 1
            for ind_h = 1:nBZ[1]-1
                append!(qh_base, qh0.+ind_h)
            end
        end

        for ind = 1:nspin
            pos = lattice.unitcell.basis[ind]
            
            phase = [exp(-im*(qh*b1*pos[1])) for qh in qh_base]

            sx_mat = lattice.spins[1,ind:nspin:end]
            sy_mat = lattice.spins[2,ind:nspin:end]
            sz_mat = lattice.spins[3,ind:nspin:end]

            str_fac_x = str_fac_x .+ repeat(fftshift(fft(sx_mat)),nBZ[1]).*phase
            str_fac_y = str_fac_y .+ repeat(fftshift(fft(sy_mat)),nBZ[1]).*phase
            str_fac_z = str_fac_z .+ repeat(fftshift(fft(sz_mat)),nBZ[1]).*phase
        end

        sf2 = real(str_fac_x.*conj(str_fac_x) + 
                str_fac_y.*conj(str_fac_y) + 
                str_fac_z.*conj(str_fac_z)) 
    end

    if D==2  # 2D lattice
        # initialize
        str_fac_x = zeros(nBZ[1]*lattice.size[1], nBZ[2]*lattice.size[2])*im
        str_fac_y = zeros(nBZ[1]*lattice.size[1], nBZ[2]*lattice.size[2])*im
        str_fac_z = zeros(nBZ[1]*lattice.size[1], nBZ[2]*lattice.size[2])*im
        sf2 = zeros(nBZ[1]*lattice.size[1], nBZ[2]*lattice.size[2])

        a1 = [lattice.unitcell.primitive[1]..., 0]
        a2 = [lattice.unitcell.primitive[2]..., 0]
        a3 = [0, 0, 1]
        vol = dot(a1, cross(a2, a3))

        b1 = 2π/vol*cross(a2,a3)
        b2 = 2π/vol*cross(a3,a1)

        # mat_p2c = [transpose([lattice.unitcell.primitive[1]...]);
        #            transpose([lattice.unitcell.primitive[2]...])]
        # mat_c2p = inv(mat_p2c)

        qh0 = collect(fftshift(fftfreq(lattice.size[1])))
        qk0 = collect(fftshift(fftfreq(lattice.size[2])))

        qh_base = copy(qh0)
        if nBZ[1] > 1
            for ind_h = 1:nBZ[1]-1
                append!(qh_base, qh0.+ind_h)
            end
        end

        qk_base = copy(qk0)
        if nBZ[2] > 1
            for ind_k = 1:nBZ[2]-1
                append!(qk_base, qk0.+ind_k)
            end
        end

        for ind = 1:nspin
            pos = lattice.unitcell.basis[ind]
            
            phase = [exp(-im*((qh*b1[1]+qk*b2[1])*pos[1] + 
                            (qh*b1[2]+qk*b2[2])*pos[2])) for qh in qh_base, qk in qk_base]

            sx_mat = reshape(lattice.spins[1,ind:nspin:end], lattice.size[2], lattice.size[1])
            sx_mat = transpose(sx_mat)
            sy_mat = reshape(lattice.spins[2,ind:nspin:end], lattice.size[2], lattice.size[1])
            sy_mat = transpose(sy_mat)
            sz_mat = reshape(lattice.spins[3,ind:nspin:end], lattice.size[2], lattice.size[1])
            sz_mat = transpose(sz_mat)

            str_fac_x = str_fac_x .+ repeat(fftshift(fft(sx_mat)),nBZ[1],nBZ[2]).*phase
            str_fac_y = str_fac_y .+ repeat(fftshift(fft(sy_mat)),nBZ[1],nBZ[2]).*phase
            str_fac_z = str_fac_z .+ repeat(fftshift(fft(sz_mat)),nBZ[1],nBZ[2]).*phase
        end

        sf2 = real(str_fac_x.*conj(str_fac_x) + 
                str_fac_y.*conj(str_fac_y) + 
                str_fac_z.*conj(str_fac_z)) 


        # chiral scattering 
#=
        # unit vectors along Q as defined in the lab system
        Q_nx = [(qh*b1[1]+qk*b2[1])/norm([qh*b1[1]+qk*b2[1], qh*b1[2]+qk*b2[2]]) for qh in qh_base, qk in qk_base]
        Q_ny = [(qh*b1[2]+qk*b2[2])/norm([qh*b1[1]+qk*b2[1], qh*b1[2]+qk*b2[2]]) for qh in qh_base, qk in qk_base]
        Q_nz = zeros(size(Q_nx))

        # Q_n \cdot str_fac
        nQ = str_fac_x.*Q_nx .+ str_fac_y.*Q_ny .+ str_fac_z.*Q_nz
        # str_fac perpendicular to Q_n
        sf_perpx = str_fac_x .- nQ.*Q_nx
        sf_perpy = str_fac_y .- nQ.*Q_ny
        sf_perpz = str_fac_z .- nQ.*Q_nz

        # pol//x
        sf_chiral = (1im*Q_nx.*(conj(sf_perpy).*sf_perpz .- conj(sf_perpz).*sf_perpy)
                  + 1im*Q_ny.*(conj(sf_perpz).*sf_perpx .- conj(sf_perpx).*sf_perpz)
                  + 1im*Q_nz.*(conj(sf_perpx).*sf_perpy .- conj(sf_perpy).*sf_perpx))
        sf_chiral = real(sf_chiral)


        # pol//z, total moment
        # sf_chiral = 1im.*(conj(str_fac_x).*str_fac_y .- conj(str_fac_y).*str_fac_x)
        # sf_chiral = real(sf_chiral)
=#

    end

    if D==3  # 3D lattice
        # initialize
        str_fac_x = zeros(nBZ[1]*lattice.size[1], nBZ[2]*lattice.size[2], nBZ[3]*lattice.size[3])*im
        str_fac_y = zeros(nBZ[1]*lattice.size[1], nBZ[2]*lattice.size[2], nBZ[3]*lattice.size[3])*im
        str_fac_z = zeros(nBZ[1]*lattice.size[1], nBZ[2]*lattice.size[2], nBZ[3]*lattice.size[3])*im
        sf2 = zeros(nBZ[1]*lattice.size[1], nBZ[2]*lattice.size[2], nBZ[3]*lattice.size[3])

        # a1 = [lattice.unitcell.primitive[1]...]
        # a2 = [lattice.unitcell.primitive[2]...]
        # a3 = [lattice.unitcell.primitive[3]...]
        # vol = dot(a1, cross(a2, a3))

        # b1 = 2π/vol*cross(a2,a3)
        # b2 = 2π/vol*cross(a3,a1)
        # b3 = 2π/vol*cross(a1,a2)

        # mat_p2c = [transpose([lattice.unitcell.primitive[1]...]);
        #            transpose([lattice.unitcell.primitive[2]...])]
        # mat_c2p = inv(mat_p2c)

        # qh0 = collect(fftshift(fftfreq(lattice.size[1])))
        # qk0 = collect(fftshift(fftfreq(lattice.size[2])))
        # ql0 = collect(fftshift(fftfreq(lattice.size[3])))

        # qh_base = Float64[]
        # for ind_h = -ceil(Int16,nBZ[1]/2)+1:floor(Int16,nBZ[1]/2)
        #     append!(qh_base, qh0.+ind_h)
        # end

        # qk_base = Float64[]
        # for ind_k = -ceil(Int16,nBZ[2]/2)+1:floor(Int16,nBZ[2]/2)
        #     append!(qk_base, qk0.+ind_k)
        # end

        # ql_base = Float64[]
        # for ind_l = -ceil(Int16,nBZ[3]/2)+1:floor(Int16,nBZ[3]/2)
        #     append!(ql_base, ql0.+ind_l)
        # end


        for ind = 1:nspin
            # pos = lattice.unitcell.basis[ind]
            
            # phase::Array{ComplexF64, 3} = [exp(-im*((qh*b1[1]+qk*b2[1]+ql*b3[1])*pos[1] + 
            #                                         (qh*b1[2]+qk*b2[2]+ql*b3[2])*pos[2] +
            #                                         (qh*b1[3]+qk*b2[3]+ql*b3[3])*pos[3]))  for qh in qh_base, qk in qk_base, ql in ql_base]

            sx_mat = reshape(lattice.spins[1,ind:nspin:end], lattice.size[3], lattice.size[2], lattice.size[1])
            sx_mat = permutedims(sx_mat, [3,2,1])
            sy_mat = reshape(lattice.spins[2,ind:nspin:end], lattice.size[3], lattice.size[2], lattice.size[1])
            sy_mat = permutedims(sy_mat, [3,2,1])
            sz_mat = reshape(lattice.spins[3,ind:nspin:end], lattice.size[3], lattice.size[2], lattice.size[1])
            sz_mat = permutedims(sz_mat, [3,2,1])

            str_fac_x = str_fac_x .+ repeat(fftshift(fft(sx_mat)),nBZ[1], nBZ[2], nBZ[3]).*lattice.phase[:,:,:,ind]
            str_fac_y = str_fac_y .+ repeat(fftshift(fft(sy_mat)),nBZ[1], nBZ[2], nBZ[3]).*lattice.phase[:,:,:,ind]
            str_fac_z = str_fac_z .+ repeat(fftshift(fft(sz_mat)),nBZ[1], nBZ[2], nBZ[3]).*lattice.phase[:,:,:,ind]
        end

        sf2 = real(str_fac_x.*conj(str_fac_x) + 
                   str_fac_y.*conj(str_fac_y) + 
                   str_fac_z.*conj(str_fac_z)) 

        
#=   chiral scattring
        # unit vectors along Q as defined in the lab system
        Q_len = [norm([qh*b1[1]+qk*b2[1]+ql*b3[1], qh*b1[2]+qk*b2[2]+ql*b3[2], qh*b1[3]+qk*b2[3]+ql*b3[3]]) 
                for qh in qh_base, qk in qk_base, ql in ql_base]
        Q_nx = [(qh*b1[1]+qk*b2[1]+ql*b3[1]) for qh in qh_base, qk in qk_base, ql in ql_base]./Q_len
        Q_ny = [(qh*b1[2]+qk*b2[2]+ql*b3[2]) for qh in qh_base, qk in qk_base, ql in ql_base]./Q_len
        Q_nz = [(qh*b1[3]+qk*b2[3]+ql*b3[3]) for qh in qh_base, qk in qk_base, ql in ql_base]./Q_len

        # Q_n \cdot str_fac
        nQ = str_fac_x.*Q_nx .+ str_fac_y.*Q_ny .+ str_fac_z.*Q_nz

        # str_fac perpendicular to Q_n
        sf_perpx = str_fac_x .- nQ.*Q_nx
        sf_perpy = str_fac_y .- nQ.*Q_ny
        sf_perpz = str_fac_z .- nQ.*Q_nz

        # pol//x
        # sf_chiral = 1im*Q_nx.*(conj(sf_perpy).*sf_perpz .- conj(sf_perpz).*sf_perpy)
        #           + 1im*Q_ny.*(conj(sf_perpz).*sf_perpx .- conj(sf_perpx).*sf_perpz)
        #           + 1im*Q_nz.*(conj(sf_perpx).*sf_perpy .- conj(sf_perpy).*sf_perpx)

        sf_chiral = (1im.*Q_nx.*(conj(str_fac_y).*str_fac_z .- conj(str_fac_z).*str_fac_y)
                  .+ 1im.*Q_ny.*(conj(str_fac_z).*str_fac_x .- conj(str_fac_x).*str_fac_z)
                  .+ 1im.*Q_nz.*(conj(str_fac_x).*str_fac_y .- conj(str_fac_y).*str_fac_x))
        
        sf_chiral = real.(sf_chiral)

        # check the sf_perp
        # sf_chiral = conj(sf_perpx).*sf_perpx .+ conj(sf_perpy).*sf_perpy .+ conj(sf_perpz).*sf_perpz
        # sf_chiral[isnan.(sf_chiral)] .= 0
        # sf_chiral = real(sf_chiral)

        # pol//z, total moment
        # sf_chiral = 1im.*(conj(str_fac_x).*str_fac_y .- conj(str_fac_y).*str_fac_x)
        # sf_chiral = real(sf_chiral)
=#

    end
    if ndims(sf2)==3
        return sf2
        # return sf_chiral
    else # for 2D and 1D lattices
        return repeat(sf2, 1,1,1)
        # return repeat(sf_chiral, 1,1,1)
    end

end

# molecular dynamics
function getDynamics(lattice::Lattice{D,N}) where {D,N}
    u0 = lattice.spins # initial condition
    if lattice.disorder == true
        rng = Random.GLOBAL_RNG
        disorderRand = rand(rng, lattice.length)
    end
    # define the evolv function
    field::Vector{Float64} = zeros(3)
    function evolv!(du::Matrix{Float64}, u::Matrix{Float64}, p, t)
        for site in 1:lattice.length
            # interactions with a designated site
            interactionSites = lattice.interactionSites[site]
            interactionMatrices = lattice.interactionMatrices[site]
            # calculate the molecular field from the interaction matrices
            field[1] = 0.0 # no applied field
            field[2] = 0.0
            field[3] = 0.0
            # field = [m.lattice.interactionField[site]...] # with applied field
            for i in 1:length(interactionSites)
                fieldx = (u[1, interactionSites[i]]*interactionMatrices[i].m11
                        + u[2, interactionSites[i]]*interactionMatrices[i].m21
                        + u[3, interactionSites[i]]*interactionMatrices[i].m31)

                fieldy = (u[1, interactionSites[i]]*interactionMatrices[i].m12
                        + u[2, interactionSites[i]]*interactionMatrices[i].m22
                        + u[3, interactionSites[i]]*interactionMatrices[i].m32)

                fieldz = (u[1, interactionSites[i]]*interactionMatrices[i].m13
                        + u[2, interactionSites[i]]*interactionMatrices[i].m23
                        + u[3, interactionSites[i]]*interactionMatrices[i].m33)

                # field += [fieldx, fieldy, fieldz]
                field[1] += fieldx
                field[2] += fieldy
                field[3] += fieldz
            end

            # add onsiteInteractions
            fieldx_Onsite = (u[1,site]*lattice.interactionOnsite[site].m11
                           + u[2,site]*lattice.interactionOnsite[site].m21
                           + u[3,site]*lattice.interactionOnsite[site].m31)

            fieldy_Onsite = (u[1,site]*lattice.interactionOnsite[site].m12
                           + u[2,site]*lattice.interactionOnsite[site].m22
                           + u[3,site]*lattice.interactionOnsite[site].m32)

            fieldz_Onsite = (u[1,site]*lattice.interactionOnsite[site].m13
                           + u[2,site]*lattice.interactionOnsite[site].m23
                           + u[3,site]*lattice.interactionOnsite[site].m33)

            # field += [fieldx_Onsite, fieldy_Onsite, fieldz_Onsite]
            field[1] += fieldx_Onsite
            field[2] += fieldy_Onsite
            field[3] += fieldz_Onsite

            if lattice.disorder == true
                field = field.*disorderRand[site]
            end

            # du[1,site], du[2,site], du[3,site] = cross(field, [u[1,site], u[2,site], u[3,site]])
            du[1,site] = field[2]*u[3,site] - field[3]*u[2,site]
            du[2,site] = field[3]*u[1,site] - field[1]*u[3,site]
            du[3,site] = field[1]*u[2,site] - field[2]*u[1,site]

        end
    end

    tau = lattice.tau
    nstep = lattice.nstep
    tspan = (0.0, tau*(nstep-1))
    prob = ODEProblem(evolv!, u0, tspan)
    # solve the differential equation
    sol = solve(prob, Vern7(), reltol=1e-2, saveat=tau)
    # transfer the solution from a vector of 2D matrices into a 3D matrix
    spins_evolv = reshape(hcat(sol.u...), (size(sol.u[1],1), size(sol.u[1],2), length(sol.u)) ) # length(sol.u) ~ nstep

    # apply a gaussian envelope
    # evlp_center = (nstep-1)/2+1
    # evlp_fwhm = evlp_center/2
    # evlp = exp.(-(collect(1:nstep) .- evlp_center).^2 ./ (evlp_fwhm^2*0.7212))
    # evlp = reshape(evlp, 1,1,length(evlp))
    # spins_evolv = spins_evolv.*evlp

    # apply a Parzen filter
    evlp_center = (nstep-1)/2+1
    parzen_x = (collect(1:nstep) .- evlp_center)./evlp_center
    evlp = zeros(length(parzen_x))
    for ind_x = 1:length(evlp)
        x = parzen_x[ind_x]
        if abs(x) <= 1/2
            evlp[ind_x] = 1-6*x^2+6*abs(x)^3
        else
            evlp[ind_x] = 2*(1-abs(x))^3
        end
    end
    evlp = reshape(evlp, 1, 1, length(evlp))
    spins_evolv = spins_evolv.*evlp

    # 2D model
    if D == 2
        # qh, qk, and omega basis for interpolation
        dBZ = lattice.dynLim
        L = lattice.size
        qh0 = collect(fftshift(fftfreq(L[1]))); 
        qk0 = collect(fftshift(fftfreq(L[2])));
        # omega_base = fftshift(fftfreq(n_timestep)).*2π./timestep
        
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

        a1 = [lattice.unitcell.primitive[1]..., 0]
        a2 = [lattice.unitcell.primitive[2]..., 0]
        a3 = [0, 0, 1]
        vol = dot(a1, cross(a2, a3))

        b1 = 2π/vol*cross(a2,a3)
        b2 = 2π/vol*cross(a3,a1)

        Sx_fourier = Sy_fourier = Sz_fourier = zeros(dBZ[1]*L[1], dBZ[2]*L[2], nstep)*im
        natom = size(lattice.unitcell.basis,1)
        for idx_atom = 1:natom
            # phase factor for each sublattice
            pos = lattice.unitcell.basis[idx_atom]
            
            phase = [exp(-im*((qh*b1[1]+qk*b2[1])*pos[1] + 
                            (qh*b1[2]+qk*b2[2])*pos[2])) for qh in qh_base, qk in qk_base] 

            # spin meshgrid for each sublattice
            Sx_mesh = reshape(spins_evolv[1,idx_atom:natom:end,:], (L[2], L[1], nstep))
            Sx_mesh = permutedims(Sx_mesh, [2,1,3])
            Sy_mesh = reshape(spins_evolv[2,idx_atom:natom:end,:], (L[2], L[1], nstep))
            Sy_mesh = permutedims(Sy_mesh, [2,1,3])
            Sz_mesh = reshape(spins_evolv[3,idx_atom:natom:end,:], (L[2], L[1], nstep))
            Sz_mesh = permutedims(Sz_mesh, [2,1,3])

            Sx_fourier = Sx_fourier + repeat(fftshift(fft(Sx_mesh)), dBZ[1], dBZ[2]).*phase
            Sy_fourier = Sy_fourier + repeat(fftshift(fft(Sy_mesh)), dBZ[1], dBZ[2]).*phase
            Sz_fourier = Sz_fourier + repeat(fftshift(fft(Sz_mesh)), dBZ[1], dBZ[2]).*phase
        end

        Sxx = Sxy = Sxz = Syx = Syy = Syz = Szx = Szy = Szz = zeros(dBZ[1]*L[1], dBZ[2]*L[2], nstep)*(0+0im)

        Sxx += Sx_fourier.*conj(Sx_fourier)
        Sxy += Sx_fourier.*conj(Sy_fourier)
        Sxz += Sx_fourier.*conj(Sz_fourier)

        Syx += Sy_fourier.*conj(Sx_fourier)
        Syy += Sy_fourier.*conj(Sy_fourier)
        Syz += Sy_fourier.*conj(Sz_fourier)

        Szx += Sz_fourier.*conj(Sx_fourier)
        Szy += Sz_fourier.*conj(Sy_fourier)
        Szz += Sz_fourier.*conj(Sz_fourier)

        Sαβ = cat(Sxx, Sxy, Sxz,
                Syx, Syy, Syz,
                Szx, Szy, Szz, dims=5)
        # swap to the dummy qL dimension
        Sαβ = permutedims(Sαβ, (1,2,4,3,5))     
        return Sαβ

    elseif D == 3
        # 3D lattice
        # qh, qk, and omega basis for interpolation
        dBZ = lattice.dynLim
        L = lattice.size
        qh0 = collect(fftshift(fftfreq(L[1])))
        qk0 = collect(fftshift(fftfreq(L[2])))
        ql0 = collect(fftshift(fftfreq(L[3])))

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

        ql_base = copy(ql0)
        if dBZ[3] > 1
            for ind_l = 1:dBZ[3]-1
                append!(ql_base, ql0.+ind_l)
            end
        end

        # omega_base = fftshift(fftfreq(n_timestep)).*2π./timestep

        a1 = [lattice.unitcell.primitive[1]...]
        a2 = [lattice.unitcell.primitive[2]...]
        a3 = [lattice.unitcell.primitive[3]...]
        vol = dot(a1, cross(a2, a3))

        b1 = 2π/vol*cross(a2,a3)
        b2 = 2π/vol*cross(a3,a1)
        b3 = 2π/vol*cross(a1,a2)

        Sx_fourier::Array{ComplexF64} = Sy_fourier::Array{ComplexF64} = Sz_fourier::Array{ComplexF64} = zeros(dBZ[1]*L[1], dBZ[2]*L[2], dBZ[3]*L[3], nstep)*im
        natom = size(lattice.unitcell.basis,1)
        for idx_atom = 1:natom
            # phase factor for each sublattice
            pos = lattice.unitcell.basis[idx_atom]
            
            phase = [exp(-im*((qh*b1[1]+qk*b2[1]+ql*b3[1])*pos[1] + 
                              (qh*b1[2]+qk*b2[2]+ql*b3[2])*pos[2] +
                              (qh*b1[3]+qk*b2[3]+ql*b3[3])*pos[3]))  for qh in qh_base, qk in qk_base, ql in ql_base]

            # spin meshgrid for each sublattice
            Sx_mesh = reshape(spins_evolv[1,idx_atom:natom:end,:], (L[3], L[2], L[1], nstep))
            Sx_mesh = permutedims(Sx_mesh, [3,2,1,4])
            Sy_mesh = reshape(spins_evolv[2,idx_atom:natom:end,:], (L[3], L[2], L[1], nstep))
            Sy_mesh = permutedims(Sy_mesh, [3,2,1,4])
            Sz_mesh = reshape(spins_evolv[3,idx_atom:natom:end,:], (L[3], L[2], L[1], nstep))
            Sz_mesh = permutedims(Sz_mesh, [3,2,1,4])

            Sx_fourier += repeat(fftshift(fft(Sx_mesh)), dBZ[1],dBZ[2],dBZ[3]).*phase
            Sy_fourier += repeat(fftshift(fft(Sy_mesh)), dBZ[1],dBZ[2],dBZ[3]).*phase
            Sz_fourier += repeat(fftshift(fft(Sz_mesh)), dBZ[1],dBZ[2],dBZ[3]).*phase
        end

        Sxx = Sxy = Sxz = Syx = Syy = Syz = Szx = Szy = Szz = zeros(dBZ[1]*L[1], dBZ[2]*L[2], dBZ[3]*L[3], nstep)*(0+0im)

        Sxx += Sx_fourier.*conj(Sx_fourier)
        Sxy += Sx_fourier.*conj(Sy_fourier)
        Sxz += Sx_fourier.*conj(Sz_fourier)

        Syx += Sy_fourier.*conj(Sx_fourier)
        Syy += Sy_fourier.*conj(Sy_fourier)
        Syz += Sy_fourier.*conj(Sz_fourier)

        Szx += Sz_fourier.*conj(Sx_fourier)
        Szy += Sz_fourier.*conj(Sy_fourier)
        Szz += Sz_fourier.*conj(Sz_fourier)

        Sαβ = cat(Sxx, Sxy, Sxz,
                Syx, Syy, Syz,
                Szx, Szy, Szz, dims=5)
        return Sαβ
    end  

end


