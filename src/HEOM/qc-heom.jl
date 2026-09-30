module QCHEOM

using OrdinaryDiffEq
using ..HEOMStructure
using ..SpectralDensities, ..Solvents, ..Utilities

function single_propagate(phasespacepoints, solvent, Hamiltonian, sops, nveclist, npluslocs, nminuslocs, mode_map, decomps, β, decay, ρ0exp, ntimes, dt, extraargs::Utilities.DiffEqArgs, verbose)
    tspan = (0.0, ntimes * dt)
    sdim = size(Hamiltonian, 1)
    ρs = zeros(ComplexF64, ntimes+1, sdim, sdim)
    workspace = zeros(ComplexF64, sdim, sdim)
    tmp1 = zeros(ComplexF64, sdim, sdim)
    sops2 = [s^2 for s in sops]
    raising_coeffs, lowering_coeffs = HEOMStructure.get_hierarchy_coeffs(nveclist, mode_map, npluslocs, nminuslocs, decomps)
    Npoints = length(phasespacepoints)
    update_len = max(1, Npoints ÷ 10)
    for (j, ps) in enumerate(phasespacepoints)
        params = HEOMStructure.HEOMParams(Hamiltonian, nothing, nothing, (solvent, ps), sops, sops2, nveclist, npluslocs, nminuslocs, mode_map, decomps, [0.0], β, decay, raising_coeffs, lowering_coeffs, workspace, tmp1)
        prob = ODEProblem{true}(HEOMStructure.scaled_HEOM_RHS!, ρ0exp, tspan, params)
        sol = solve(prob, extraargs.solver, reltol=extraargs.reltol, abstol=extraargs.abstol, saveat=dt)
        for t=1:length(sol)
            @inbounds ρs[t, :, :] .+= sol.u[t][:, :, 1]
        end
        if verbose && (j % update_len == 0)
            @info "Initial condition number $j of $(Npoints) done."
        end
    end
    ρs, Npoints
end

function propagate(; Hamiltonian::Matrix{ComplexF64}, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, solvent::Solvents.Solvent, ρ0::Matrix{ComplexF64}, β::Real, dt::Real, ntimes::Int, tol::Float64, Lmax::Int, sops::Vector{Matrix{ComplexF64}}, extraargs::Utilities.DiffEqArgs=Utilities.DiffEqArgs(), verbose::Bool=false)
    nbaths = length(Jw)
    decomps = Vector{SpectralDensityDecompositions.ExponentialDecomposition}(undef, length(Jw))
    for (i, jw) in enumerate(Jw)
        # @assert typeof(jw) == SpectralDensities.DrudeLorentz "HEOM has only been implemented for the Drude-Lorentz spectral density."
        decomps[i] = SpectralDensityDecompositions.decompose(jw, tol)
        @info "Decomposed bath number $i."
    end
    nveclist, npluslocs, nminuslocs, mode_map = HEOMStructure.setup_simulation(decomps, Lmax)
    @info "Setup complete. Starting run"
    @info "Number of ADOs used: $(length(nveclist))"

    decay = HEOMStructure.get_decay(nveclist, mode_map, decomps)

    sdim = size(ρ0, 1)
    Nh = length(nveclist)
    ρ0exp = zeros(ComplexF64, sdim, sdim, Nh)
    ρ0exp[:, :, 1] .= ρ0
    chunks = Iterators.partition(solvent, cld(length(solvent), Threads.nthreads()))
    ρtasks = map(enumerate(chunks)) do (ind, chunk)
        Threads.@spawn single_propagate(chunk, solvent, Hamiltonian, sops, nveclist, npluslocs, nminuslocs, mode_map, decomps, β, decay, copy(ρ0exp), ntimes, dt, extraargs, verbose && (ind==1))
    end
    results = fetch.(ρtasks)
    ρs = zero(results[1][1])
    nsamples = 0
    for (res, j) in results
        ρs .+= res
        nsamples += j
    end

    0:dt:ntimes*dt, ρs/nsamples
end

end
