module HEOM

using HDF5
using OrdinaryDiffEq
using ..HEOMStructure
using ..SpectralDensities, ..SpectralDensityDecompositions, ..Solvents, ..Utilities

const references = """
- Y. Tanimura and R. Kubo, Time Evolution of a Quantum System in Contact with a Nearly Gaussian-Markoffian Noise Bath, Journal of the Physical Society of Japan 58, 101 (1989).
- Q. Shi, L. Chen, G. Nan, R.-X. Xu, and Y. Yan, Efficient hierarchical Liouville space propagator to quantum dissipative dynamics, J. Chem. Phys. 130, 084105 (2009)."""


"""
    propagate(; Hamiltonian::AbstractMatrix{ComplexF64}, ρ0::AbstractMatrix{ComplexF64}, β::Real, Jw::AbstractVector{SpectralDensities.SpectralDensity}, sys_ops::Vector{Matrix{ComplexF64}}, num_modes::Int, Lmax::Int, dt::Real, ntimes::Int, threshold::Float64=0.0, scaled::Bool=true, L::Union{Nothing,Vector{Matrix{ComplexF64}}}=nothing, external_fields::Union{Nothing,Vector{Utilities.ExternalField}}=nothing, extraargs::Utilities.DiffEqArgs=Utilities.DiffEqArgs())
Uses HEOM to propagate the initial reduced density matrix, `ρ0`, under the given `Hamiltonian`, and set of spectral densities, `Jw`, interacting with the system through `sys_ops`.

- `ρ0`: initial reduced density matrix
- `Hamiltonian`: system Hamiltonian
- `external_fields`: either `nothing` or a vector of external time-dependent fields
- `Jw`: array of spectral densities
- `sys_ops`: system operators through which the corresponding baths interact
- `L`: vector of Lindblad jump operators

- `num_modes`: number of Matsubara modes to be considered
- `Lmax`: cutoff for maximum number of levels
- `dt`: time-step for recording the density matrices
- `ntimes`: number of time steps of simulation
- `threshold`: filtration threshold
- `extraargs`: extra arguments for the differential equation solver
"""
function propagate(; Hamiltonian::AbstractMatrix{ComplexF64}, ρ0::AbstractMatrix{ComplexF64}, β::Real, Jw::AbstractVector{SpectralDensities.SpectralDensity}, sys_ops::Vector{Matrix{ComplexF64}}, tol::Float64, num_modes::Int, Lmax::Int, dt::Real, ntimes::Int, threshold::Float64=0.0, L::Union{Nothing,Vector{Matrix{ComplexF64}}}=nothing, external_fields::Union{Nothing,Vector{Utilities.ExternalField}}=nothing, extraargs::Utilities.DiffEqArgs=Utilities.DiffEqArgs(), decomposition::String, verbose=false, separable=true, output::Union{Nothing,HDF5.Group}=nothing)
    decomps = Vector{SpectralDensityDecompositions.ExponentialDecomposition}(undef, length(Jw))
    Δk = zeros(length(Jw))
    Δk_imag = zeros(length(Jw))
    for (i, jw) in enumerate(Jw)
        # decomps[i] = decomposition == "matsubara" ? SpectralDensityDecompositions.matsubara_decomposition(jw, num_modes, β) : SpectralDensityDecompositions.pade_decomposition(jw, num_modes, β)
        decomps[i], Δk_target = SpectralDensityDecompositions.thermal_decomposition(jw, tol, num_modes, decomposition, β)
        tmp = sum(decomps[i].c ./ decomps[i].ν)
        Δk[i] = (Δk_target - real(tmp)) # residual sum used to truncate the hierarchy
        Δk_imag[i] = (-SpectralDensities.reorganization_energy(jw)/jw.Δs^2 - imag(tmp))
        verbose && @info "Decomposed bath number $i."
    end
    nveclist, npluslocs, nminuslocs, mode_map = HEOMStructure.setup_simulation(decomps, Lmax)
    verbose && @info "Setup complete. Starting run"
    @info "Number of ADOs used: $(length(nveclist))"

    H = deepcopy(Hamiltonian)
    for (Δi, co) in zip(Δk_imag, sys_ops)
        H .+= Δi * (co * co)
    end

    Nh = length(nveclist)
    sdim = size(ρ0, 1)
    workspace = zeros(ComplexF64, sdim, sdim)
    tmp1 = zeros(ComplexF64, sdim, sdim)

    LdagL = if isnothing(L)
        nothing
    else
        [l' * l for l in L]
    end
    decay = HEOMStructure.get_decay(nveclist, mode_map, decomps)
    sys_ops2 = [s^2 for s in sys_ops]
    raising_coeffs, lowering_coeffs = HEOMStructure.get_hierarchy_coeffs(nveclist, mode_map, npluslocs, nminuslocs, decomps)
    params = HEOMStructure.HEOMParams(H, L, LdagL, external_fields, sys_ops, sys_ops2, nveclist, npluslocs, nminuslocs, mode_map, decomps, Δk, β, decay, raising_coeffs, lowering_coeffs, workspace, tmp1)
    tspan = (0.0, dt * ntimes)
    sdim = size(ρ0, 1)
    ρ0_expanded = zeros(ComplexF64, sdim, sdim, Nh)
    if separable
        ρ0_expanded[:, :, 1] .= ρ0
    end
    ts = 0:dt:(ntimes*dt)
    ρs = zeros(ComplexF64, length(ts), sdim, sdim)
    ρs[1, :, :] .= ρ0

    if !isnothing(output)
        Utilities.check_or_insert_value(output, "rho", ρs)
        Utilities.check_or_insert_value(output, "time_taken", zeros(Float64, ntimes))
    end
    prob = ODEProblem{true}(HEOMStructure.scaled_HEOM_RHS!, ρ0_expanded, tspan, params)

    integ = init(prob, extraargs.solver; reltol=extraargs.reltol, abstol=extraargs.abstol, dense=false, calck=false)
    k = 2
    for t in ts[2:end]
        # step_time = @elapsed step!(integ, dt, true)
        _, time_taken, memory_allocated, gc_time, _ = @timed begin
            step!(integ, dt, true)
        end
        @inbounds @views ρs[k, :, :] .= integ.u[:,:,1]
        verbose && @info "Step = $(k-1); time = $(round(time_taken; digits=3)) sec; memory allocated = $(round(memory_allocated / 1024^3; digits=3)) GiB; gc time = $(round(gc_time; digits=3)) sec"
        if !isnothing(output)
            output["rho"][k, :, :] = ρs[k, :, :]
            output["time_taken"][k-1] = time_taken
            flush(output)
        end
        k += 1
    end
    ts, ρs
end

end
