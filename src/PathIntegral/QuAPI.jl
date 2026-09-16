module QuAPI

using LinearAlgebra
using HDF5
using FLoops
using ..EtaCoefficients, ..Propagators, ..SpectralDensities, ..Utilities, ..TTM

const references = """
- Makri, N.; Makarov, D. E. Tensor Propagator for Iterative Quantum Time Evolution of Reduced Density Matrices. I. Theory. The Journal of Chemical Physics 1995, 102 (11), 4600–4610. https://doi.org/10.1063/1.469509.
- Makri, N.; Makarov, D. E. Tensor Propagator for Iterative Quantum Time Evolution of Reduced Density Matrices. II. Numerical Methodology. The Journal of Chemical Physics 1995, 102 (11), 4611–4618. https://doi.org/10.1063/1.469508."""

struct States
    sbar::Matrix{Float64}
    Δs::Matrix{Float64}
    forward_ind::Vector{UInt64}
    backward_ind::Vector{UInt64}
end

struct Path{T}
    states::Vector{UInt64}
    amplitude::T
    sdim2::UInt64
end
function next_path(p::Path{T}) where {T<:Number}
    states = p.states
    sdim2 = p.sdim2
    for j = 1:length(states)
        if states[j] < sdim2
            states[j] += 1
            break
        elseif states[j] == sdim2
            states[j] = 1
        end
    end
    return Path{T}(states, zero(T), sdim2), states != repeat([sdim2], length(states))
end

@inbounds function setup_simulation(ρ0, η::AbstractVector{EtaCoefficients.EtaCoeffs}, svec, cutoff)
    sdim = size(ρ0, 1)
    sdim_square = sdim^2
    nbaths = size(svec, 1)
    sbar = zeros(nbaths, sdim_square)
    Δs = zeros(nbaths, sdim_square)
    forward_ind = zeros(UInt64, sdim_square)
    backward_ind = zeros(UInt64, sdim_square)
    amplitudes = zeros(ComplexF64, sdim_square)
    count = 1
    for forind = 1:size(svec, 2)
        for backind = 1:size(svec, 2)
            forward_ind[count] = forind
            backward_ind[count] = backind
            amplitudes[count] = ρ0[forind, backind]
            count += 1
        end
    end
    for (bath_num, bath_svec) in enumerate(eachrow(svec))
        count = 1
        for sf in bath_svec
            for sb in bath_svec
                sbar[bath_num, count] = (sf + sb) / 2.0
                Δs[bath_num, count] = sf - sb
                count += 1
            end
        end
    end
    state_values = States(sbar, Δs, forward_ind, backward_ind)
    paths = Vector{Path{ComplexF64}}()
    for count = 1:sdim_square
        states = [count]
        if abs(amplitudes[count]) > cutoff.cutoff
            amp = amplitudes[count]
            for (bn, bη) in enumerate(η)
                amp *= exp(-state_values.Δs[bn, count] * (real(bη.η00) * state_values.Δs[bn, count] + 2im * imag(bη.η00) * state_values.sbar[bn, count]))
            end
            push!(paths, Path{ComplexF64}(states, amp, sdim_square))
        end
    end
    state_values, paths
end

@inbounds function setup_simulation(ρ0, ζ::AbstractVector{EtaCoefficients.ZetaCoeffs}, svec, cutoff, srefs)
    sdim = size(ρ0, 1)
    sdim_square = sdim^2
    nbaths = size(svec, 1)
    sbar = zeros(nbaths, sdim_square)
    Δs = zeros(nbaths, sdim_square)
    forward_ind = zeros(UInt64, sdim_square)
    backward_ind = zeros(UInt64, sdim_square)
    amplitudes = zeros(ComplexF64, sdim_square)
    count = 1
    for forind = 1:size(svec, 2)
        for backind = 1:size(svec, 2)
            forward_ind[count] = forind
            backward_ind[count] = backind
            amplitudes[count] = ρ0[forind, backind]
            count += 1
        end
    end
    for (bath_num, bath_svec) in enumerate(eachrow(svec))
        count = 1
        for sf in bath_svec
            for sb in bath_svec
                sbar[bath_num, count] = (sf + sb) / 2.0
                Δs[bath_num, count] = sf - sb
                count += 1
            end
        end
    end
    state_values = States(sbar, Δs, forward_ind, backward_ind)
    paths = Vector{Path{ComplexF64}}()
    for count = 1:sdim_square
        states = [count]
        if abs(amplitudes[count]) > cutoff.cutoff
            amp = amplitudes[count]
            for (bn, bζ) in enumerate(ζ)
                amp *= exp(1im * state_values.Δs[bn, count] * bζ.ζ00 * (state_values.sbar[bn, count] - srefs[bn]))
            end
            push!(paths, Path{ComplexF64}(states, amp, sdim_square))
        end
    end
    state_values, paths
end

function get_influence(η::EtaCoefficients.EtaCoeffs, bath_number::Int, state_values::States, path::Vector{UInt64}, terminal::Bool, in_memory::Bool)
    @inbounds begin
        Δsfinal = state_values.Δs[bath_number, path[end]]
        if Δsfinal == 0
            return 1
        end
        num_time = length(path)
        if in_memory
            if terminal
                influence_exponent = real(η.η00) * Δsfinal + 2im * imag(η.η00) * state_values.sbar[bath_number, path[end]]
                for i = 2:num_time-1
                    influence_exponent += real(η.η0m[num_time-i]) * state_values.Δs[bath_number, path[i]] + 2im * imag(η.η0m[num_time-i]) * state_values.sbar[bath_number, path[i]]
                end
                influence_exponent += real(η.η0e[num_time-1]) * state_values.Δs[bath_number, path[1]] + 2im * imag(η.η0e[num_time-1]) * state_values.sbar[bath_number, path[1]]
                return exp(-Δsfinal * influence_exponent)
            else
                influence_exponent = real(η.ηmm) * Δsfinal + 2im * imag(η.ηmm) * state_values.sbar[bath_number, path[end]]
                for i = 2:num_time-1
                    influence_exponent += real(η.ηmn[num_time-i]) * state_values.Δs[bath_number, path[i]] + 2im * imag(η.ηmn[num_time-i]) * state_values.sbar[bath_number, path[i]]
                end
                influence_exponent += real(η.η0m[num_time-1]) * state_values.Δs[bath_number, path[1]] + 2im * imag(η.η0m[num_time-1]) * state_values.sbar[bath_number, path[1]]
                return exp(-Δsfinal * influence_exponent)
            end
        else
            if terminal
                influence_exponent = real(η.η00) * Δsfinal + 2im * imag(η.η00) * state_values.sbar[bath_number, path[end]]
                for i = 1:num_time-1
                    influence_exponent += real(η.η0m[num_time-i]) * state_values.Δs[bath_number, path[i]] + 2im * imag(η.η0m[num_time-i]) * state_values.sbar[bath_number, path[i]]
                end
                return exp(-Δsfinal * influence_exponent)
            else
                influence_exponent = real(η.ηmm) * Δsfinal + 2im * imag(η.ηmm) * state_values.sbar[bath_number, path[end]]
                for i = 1:num_time-1
                    influence_exponent += real(η.ηmn[num_time-i]) * state_values.Δs[bath_number, path[i]] + 2im * imag(η.ηmn[num_time-i]) * state_values.sbar[bath_number, path[i]]
                end
                return exp(-Δsfinal * influence_exponent)
            end
        end
    end
end

function get_influence(ζ::EtaCoefficients.ZetaCoeffs, bath_number::Int, state_values::States, path::Vector{UInt64}, srefs::Vector{Float64}, terminal::Bool, in_memory::Bool, tindex::Int)
    @inbounds begin
        Δsfinal = state_values.Δs[bath_number, path[end]]
        if Δsfinal == 0
            return 1
        end
        num_time = length(path)
        sbar_diff_next = state_values.sbar[bath_number, path[1:end-1]] .- srefs[tindex-num_time+2:tindex]
        # sbar_diff_next = terminal ? state_values.sbar[bath_number, path[1:end-1]] .- srefs[tindex-num_time+2:tindex] : state_values.sbar[bath_number, path] .- srefs[tindex-num_time+2:tindex+1]
        sbar_diff_prev = in_memory ? state_values.sbar[bath_number, path[2:end]] .- srefs[1:tindex] : state_values.sbar[bath_number, path[1:end]] .- srefs[max(tindex-num_time+1,1):tindex]
        if in_memory
            @assert num_time == tindex+1
            if terminal
                influence_exponent = ζ.ζ00 * sbar_diff_prev[end]
                for i = 2:num_time-1
                    influence_exponent += ζ.ζme[num_time-i, 1] * sbar_diff_prev[i]
                    influence_exponent += ζ.ζme[num_time-i, 2] * sbar_diff_next[i]
                end
                influence_exponent += ζ.ζ0e[num_time-1] * sbar_diff_next[1]
                return exp(1im * Δsfinal * influence_exponent + 1im * ζ.ζmm[2] * state_values.Δs[bath_number, path[end-1]] * sbar_diff_next[end])
            else
                influence_exponent = ζ.ζmm[1] * sbar_diff_prev[end]
                # influence_exponent += ζ.ζmm[2] * sbar_diff_next[end]
                for i = 2:num_time-1
                    influence_exponent += ζ.ζmn[num_time-i, 1] * sbar_diff_prev[i]
                    influence_exponent += ζ.ζmn[num_time-i, 2] * sbar_diff_next[i]
                end
                influence_exponent += ζ.ζ0m[num_time-1] * sbar_diff_next[1]
                return exp(1im * Δsfinal * influence_exponent)
            end
        else
            if terminal
                influence_exponent = ζ.ζ00 * sbar_diff_prev[end]
                for i = 1:num_time-1
                    influence_exponent += ζ.ζme[num_time-i, 1] * sbar_diff_prev[i]
                    influence_exponent += ζ.ζme[num_time-i, 2] * sbar_diff_next[i]
                end
                return exp(1im * Δsfinal * influence_exponent + 1im * ζ.ζmm[2] * state_values.Δs[bath_number, path[end-1]] * sbar_diff_next[end])
                # return exp(1im * Δsfinal * influence_exponent)
            else
                influence_exponent = ζ.ζmm[1] * sbar_diff_prev[end] 
                # influence_exponent += ζ.ζmm[2] * sbar_diff_next[end]
                for i = 1:num_time-1
                    influence_exponent += ζ.ζmn[num_time-i, 1] * sbar_diff_prev[i]
                    influence_exponent += ζ.ζmn[num_time-i, 2] * sbar_diff_next[i]
                end
                return exp(1im * Δsfinal * influence_exponent)
            end
        end
    end
end

get_influence(η::AbstractVector{EtaCoefficients.EtaCoeffs}, state_values::States, path::Vector{UInt64}, srefs::AbstractMatrix{Float64}, terminal::Bool, in_memory::Bool, tindex::Int) = prod([get_influence(bη, bn, state_values, path, terminal, in_memory) for (bn, bη) in enumerate(η)])
get_influence(ζ::AbstractVector{EtaCoefficients.ZetaCoeffs}, state_values::States, path::Vector{UInt64}, srefs::AbstractMatrix{Float64}, terminal::Bool, in_memory::Bool, tindex::Int) = prod([get_influence(bζ, bn, state_values, path, srefs[:, bn], terminal, in_memory, tindex) for (bn, bζ) in enumerate(ζ)])

"""
Filtration parameters for different versions of QuAPI. Currently has a threshold for magnitude-based filtering, with a default value of `cutoff=0` (no filtering).

Properties:
- `cutoff`: threshold for absolute magnitude-based path filtration [default = 0.0]
- `prop_cutoff`: propagator cutoff threshold for adaptive kink filtration [default = 0.0]
- `num_kinks`: maximum number of kinks for kink summation algorithm [default = -1: all kinks are allowed]
- `num_blips`: maximum number of blips between forward and backward paths allowed [default = -1: all blips are allowed]
"""
struct QuAPIArgs <: Utilities.ExtraArgs
    cutoff::Float64
    prop_cutoff::Float64
    num_kinks::Int64
    num_blips::Int64
end
QuAPIArgs(; cutoff=0.0, prop_cutoff=0.0, num_kinks=-1, num_blips=-1) = QuAPIArgs(cutoff, prop_cutoff, num_kinks, num_blips)

"""
    propagate(; fbU::Matrix{ComplexF64}, Jw::Vector{T}, β::Real, ρ0, dt::Real, ntimes::Int, kmax::Int, extraargs::QuAPIArgs=QuAPIArgs(), svec=[1.0 -1.0], reference_prop=false, verbose::Bool=false) where {T<:SpectralDensities.SpectralDensity}

Given a time-series of system forward-backward propagators, `fbU`, the spectral densities describing the solvent, `Jw`, and an inverse temperature, this uses QuAPI to propagate the input initial reduced density matrix, ρ0, with a time-step of `dt` for `ntimes` time steps. A non-Markovian memory of `kmax` steps is used in this simulation. The j^th bath, described by `Jw[j]`, interacts with the system through the diagonal operator with the values of `svec[j,:]`.

Arguments:
- `ρ0`: initial reduced density matrix
- `fbU`: time-series of forward-backward propagators
- `Jw`: array of spectral densities
- `svec`: diagonal elements of system operators through which the corresponding baths interact. QuAPI currently only works for baths with diagonal coupling to the system

- `dt`: time-step for recording the density matrices
- `ntimes`: number of time steps of simulation
- `kmax`: number of steps within memory
- `extraargs`: extra arguments for the QuAPI algorithm. Contains the filtration cutoff threshold
"""
function propagate(; fbU::Union{Nothing, AbstractArray{ComplexF64,3}}=nothing, Jw::Union{Nothing, AbstractVector{T}}=nothing, η::Union{Nothing, AbstractVector{<:EtaCoefficients.IFCoeffs}}=nothing, β::Union{Nothing, Real}=nothing, ρ0::AbstractMatrix{ComplexF64}, dt::Real, ntimes::Int, kmax::Int, extraargs::QuAPIArgs=QuAPIArgs(), svec=[1.0 -1.0], verbose::Bool=false, output::Union{Nothing,HDF5.Group}=nothing, kwargs...) where {T<:SpectralDensities.SpectralDensity}
    if kmax > ntimes
        kmax = ntimes + 2
    end
    if isnothing(η)
        @assert !isnothing(Jw)
        @assert length(Jw) == size(svec, 1)
        η = isnothing(β) ? [EtaCoefficients.calculate_ζ(jw; dt, kmax) for jw in Jw] : [EtaCoefficients.calculate_η(jw; β, dt, kmax) for jw in Jw]
    end
    @assert length(η) == size(svec, 1)
    nbath = length(η)
    sdim = size(ρ0, 1)
    ρs = zeros(ComplexF64, ntimes + 1, sdim, sdim)
    ρs[1, :, :] = ρ0
    if !isnothing(output)
        Utilities.check_or_insert_value(output, "rho", ρs)
        # Utilities.check_or_insert_value(output, "time_taken", zeros(Float64, ntimes))
        Utilities.check_or_insert_value(output, "num_paths", zeros(Int64, ntimes))
    end
    state_values, paths = typeof(η)==Vector{EtaCoefficients.EtaCoeffs} ? setup_simulation(ρ0, η, svec, extraargs) : setup_simulation(ρ0, η, svec, extraargs, Propagators.get_reference(Propagators.ReferenceChoice(kwargs[:reference_choice])(), ρ0, kwargs[:solvent]))
    eacp = false
    if kmax == 0
        kmax += 1
        eacp = true
    end

    sdim2 = sdim^2
    if verbose
        @info "Starting propagation within memory"
    end
    tmprho = zeros(ComplexF64, sdim2)
    tmpfbU = zeros(ComplexF64, sdim2, sdim2)
    srefs = zeros(ntimes+1, nbath)
    if isnothing(fbU) && typeof(η) == Vector{EtaCoefficients.ZetaCoeffs}
        ps = kwargs[:ps]
        solvent = kwargs[:solvent]
        Hamiltonian = kwargs[:Hamiltonian]
        classical_dt = kwargs[:classical_dt]
        reference_choice = kwargs[:reference_choice]
    end
    for i = 1:min(kmax, ntimes)
        if verbose
            @info "Step = $(i), #paths = $(length(paths))"
        end
        if isnothing(fbU) && typeof(η) == Vector{EtaCoefficients.ZetaCoeffs}
            U, sref, ps = Propagators.calculate_reference_propagators(; Hamiltonian, solvent, ps, classical_dt, ρ0=ρs[i, :, :], dt, ntimes=1, reference_choice)
            tmpfbU .= U[1, :, :]
            srefs[i, :] = sref[1, :]
        else
            tmpfbU .= fbU[i, :, :]
            srefs[i, :] = zeros(1, nbath)
        end
        new_paths = Vector{Path{ComplexF64}}()
        for path in paths
            states = path.states
            tmprho .= zero(ComplexF64)
            @inbounds tmprho[states[end]] = 1.0
            tmprho = tmpfbU * tmprho * path.amplitude
            for (s, amp) in enumerate(tmprho)
                tmpstates = deepcopy(states)
                push!(tmpstates, s)
                amplitude = amp * get_influence(η, state_values, tmpstates, srefs[1:i, :], false, true, i)
                if eacp
                    @inbounds ρs[i+1, state_values.forward_ind[s], state_values.backward_ind[s]] += amp
                    amplitude = amp
                else
                    @inbounds ρs[i+1, state_values.forward_ind[s], state_values.backward_ind[s]] += amp * get_influence(η, state_values, tmpstates, srefs[1:i, :], true, true, i)
                end
                if abs(amplitude) > extraargs.cutoff
                    push!(new_paths, Path{ComplexF64}(tmpstates, amplitude, sdim2))
                end
            end
        end
        paths = new_paths
        new_paths = nothing
        if !isnothing(output)
            output["rho"][i+1, :, :] = ρs[i+1, :, :]
            # output["time_taken"][i] = timetaken[i]
            output["num_paths"][i] = length(paths)
            flush(output)
        end
    end

    if verbose
        @info "Starting iteration"
    end
    path_max = repeat(Vector{UInt64}([sdim2]), length(paths[1].states) - 1)
    max_length = Utilities.hash_path(path_max, sdim2)
    checked = repeat([false], max_length)
    tmat = zeros(ComplexF64, max_length)
    for i = kmax+1:ntimes
        if verbose
            @info "Step = $(i), #paths = $(length(paths))"
        end
        if isnothing(fbU)
            U, sref, ps = Propagators.calculate_reference_propagators(; Hamiltonian, solvent, ps, classical_dt, ρ0=ρs[i, :, :], dt, ntimes=1, reference_choice)
            tmpfbU .= U[1, :, :]
            srefs[i, :] = sref[1, :]
        else
            tmpfbU .= fbU[i, :, :]
            srefs[i, :] = zeros(1, nbath)
        end
        tmat .= zero(ComplexF64)
        for path in paths
            trunc_path = path.states[2:end]
            @inbounds tmat[Utilities.hash_path(trunc_path, sdim2)] += path.amplitude
        end
        new_paths = Vector{Path{ComplexF64}}()
        checked .= false
        for path in paths
            @inbounds states = path.states[2:end]
            hash_val = Utilities.hash_path(states, sdim2)
            if checked[hash_val]
                continue
            else
                tmprho .= zero(ComplexF64)
                @inbounds tmprho[states[end]] = 1.0
                tmprho = tmpfbU * tmprho * tmat[hash_val]
                for (s, amp) in enumerate(tmprho)
                    tmpstates = deepcopy(states)
                    push!(tmpstates, s)
                    amplitude = amp * get_influence(η, state_values, tmpstates, srefs[1:i, :], false, false, i)
                    if eacp
                        @inbounds ρs[i+1, state_values.forward_ind[s], state_values.backward_ind[s]] += amp
                        amplitude = amp
                    else
                        @inbounds ρs[i+1, state_values.forward_ind[s], state_values.backward_ind[s]] += amp * get_influence(η, state_values, tmpstates, srefs[1:i, :], true, false, i)
                    end
                    if abs(amplitude) > extraargs.cutoff
                        push!(new_paths, Path{ComplexF64}(tmpstates, amplitude, sdim2))
                    end
                end
                @inbounds checked[hash_val] = true
            end
        end
        paths = new_paths
        new_paths = nothing
        if !isnothing(output)
            output["rho"][i+1, :, :] = ρs[i+1, :, :]
            # output["time_taken"][i] = timetaken[i]
            output["num_paths"][i] = length(paths)
            flush(output)
        end
    end
    0:dt:ntimes*dt, ρs
end

function get_path_influence(η::EtaCoefficients.EtaCoeffs, bath_number::Int, state_values, path, quapi=true)
    @inbounds begin
        interm_influence = zero(ComplexF64)
        i = length(path) - 1
        for sk = 2:i
            if state_values.Δs[bath_number, path[sk]] != 0
                val = real(η.ηmm) * state_values.Δs[bath_number, path[sk]] + 2im * imag(η.ηmm) * state_values.sbar[bath_number, path[sk]]
                for skp = 2:sk-1
                    val += real(η.ηmn[sk-skp]) * state_values.Δs[bath_number, path[skp]] + 2im * imag(η.ηmn[sk-skp]) * state_values.sbar[bath_number, path[skp]]
                end
                interm_influence += -state_values.Δs[bath_number, path[sk]] * val
            end
        end
        amplitude = exp(interm_influence)

        init_influence_0 = -state_values.Δs[bath_number, path[1]] * (real(η.η00) * state_values.Δs[bath_number, path[1]] + 2im * imag(η.η00) * state_values.sbar[bath_number, path[1]])
        for j = 2:i
            if state_values.Δs[bath_number, path[j]] != 0
                init_influence_0 += -state_values.Δs[bath_number, path[j]] * (real(η.η0m[j-1] * state_values.Δs[bath_number, path[1]]) + 2im * imag(η.η0m[j-1]) * state_values.sbar[bath_number, path[1]])
            end
        end
        init_influence_0 = exp(init_influence_0)

        final_influence_0 = one(ComplexF64)
        term_influence_0e = one(ComplexF64)
        if state_values.Δs[bath_number, path[end]] != 0
            final_influence_0 = -state_values.Δs[bath_number, path[end]] * (real(η.η00) * state_values.Δs[bath_number, path[end]] + 2im * imag(η.η00) * state_values.sbar[bath_number, path[end]])
            for j = 2:i
                final_influence_0 += -state_values.Δs[bath_number, path[end]] * (real(η.η0m[i+1-j] * state_values.Δs[bath_number, path[j]]) + 2im * imag(η.η0m[i+1-j]) * state_values.sbar[bath_number, path[j]])
            end
            final_influence_0 = exp(final_influence_0)

            term_influence_0e = exp(-state_values.Δs[bath_number, path[end]] * (real(η.η0e[i]) * state_values.Δs[bath_number, path[1]] + 2im * imag(η.η0e[i]) * state_values.sbar[bath_number, path[1]]))
        end

        amplitude * init_influence_0 * final_influence_0 * term_influence_0e
    end
end

"""
    build_augmented_propagator(; fbU::AbstractArray{ComplexF64,3}, Jw::Vector{T}, β::Real, dt::Real, ntimes::Int, kmax::Union{Int,Nothing}=nothing, extraargs::QuAPIArgs=QuAPIArgs(), svec=[1.0 -1.0], reference_prop=false, verbose::Bool=false, output::Union{Nothing,HDF5.Group}=nothing, from_TTM::Bool=false, exec=ThreadedEx()) where {T<:SpectralDensities.SpectralDensity}

Builds the propagators, augmented with the influence of the harmonic baths defined by the spectral densities `Jw`,  upto `ntimes` time-steps without iteration using shared memory parallelism. The paths are generated in full forward-backward space but not stored. So, while the space requirement is minimal and constant, the time complexity for each time-step grows by an additional factor of ``d^2``, where ``d`` is the dimensionality of the system. This j^th bath, described by `Jw[j]`, interacts with the system through the diagonal operator with the values of `svec[j,:]`. Note that the `kmax` variable does not have any function here. All calculations are full path.

Arguments:
- `fbU`: time-series of forward-backward propagators
- `Jw`: array of spectral densities
- `svec`: diagonal elements of system operators through which the corresponding baths interact. QuAPI currently only works for baths with diagonal coupling to the system.
- `β`: inverse temperature
- `dt`: time-step for recording the density matrices
- `ntimes`: number of time steps of simulation
- `extraargs`: extra arguments for the blip algorithm. Contains the `max_blips` threshold for number of blips, and the `num_changes` threshold on the number of blip-to-blip changes.
- `exec`: FLoops.jl execution policy
"""
function build_augmented_propagator(; fbU::AbstractArray{ComplexF64,3}, Jw::Vector{T}, β::Real, dt::Real, ntimes::Int, kmax::Union{Int,Nothing}=nothing, extraargs::QuAPIArgs=QuAPIArgs(), svec=[1.0 -1.0], reference_prop=false, verbose::Bool=false, output::Union{Nothing,HDF5.Group}=nothing, from_TTM::Bool=false, exec=ThreadedEx()) where {T<:SpectralDensities.SpectralDensity}
    @assert length(Jw) == size(svec, 1)
    nb = length(Jw)
    η = [EtaCoefficients.calculate_η(jw; β, dt, kmax=ntimes, imaginary_only=reference_prop) for jw in Jw]
    sdim2 = size(fbU, 2)
    sdim = trunc(Int, sqrt(sdim2))
    state_values, _ = setup_simulation(ones(sdim, sdim), η, svec, extraargs)

    if verbose
        @info "Starting propagation within memory"
    end
    U0e = Array{ComplexF64}(undef, ntimes, sdim2, sdim2)
    T0e = zero(U0e)
    if !isnothing(output)
        if !from_TTM
            Utilities.check_or_insert_value(output, "U0e", U0e)
        end
        Utilities.check_or_insert_value(output, "T0e", T0e)
        Utilities.check_or_insert_value(output, "time_taken", zeros(Float64, ntimes))
        Utilities.check_or_insert_value(output, "num_paths", zeros(Int64, ntimes))
    end
    for i = 1:ntimes
        if verbose
            @info "Starting time step $(i)"
        end
        _, time_taken, memory_allocated, gc_time, _ = @timed begin
            @floop exec for path_num = 1:sdim2^(i+1)
                @init states = ones(UInt64, i + 1)
                Utilities.unhash_path(path_num, states, sdim2)
                bare_amplitude = one(ComplexF64)
                @inbounds for j = 1:i
                    @inbounds bare_amplitude *= fbU[j, states[j+1], states[j]]
                end
                if abs(bare_amplitude) < extraargs.cutoff
                    continue
                end
                @reduce num_paths = 0 + 1
                for bn = 1:nb
                    @inbounds bare_amplitude *= get_path_influence(η[bn], bn, state_values, states, false)
                end
                @init tmpval = zeros(ComplexF64, sdim2, sdim2)
                fill!(tmpval, 0.0+0.0im)
                @inbounds tmpval[states[end], states[1]] = bare_amplitude
                @reduce tmpU0e = zeros(ComplexF64, sdim2, sdim2) .+ tmpval
            end
            @inbounds U0e[i, :, :] .= tmpU0e
            TTM.update_Ts!(T0e, U0e, i)
        end
        if !isnothing(output)
            output["U0e"][i, :, :] = U0e[i, :, :]
            output["T0e"][i, :, :] = T0e[i, :, :]
            output["time_taken"][i] = time_taken
            output["num_paths"][i] = num_paths
            flush(output)
        end
        if verbose
            @info "Done time step $(i); # paths = $(sum(num_paths)); norm(Tmat) = $(round(norm(T0e[i, :, :]); digits=3)); time = $(round(time_taken; digits=3)) sec; memory allocated = $(round(memory_allocated / 1024^3; digits=3)) GiB; gc time = $(round(gc_time; digits=3)) sec"
        end
    end
    U0e
end

function build_augmented_propagator_kink(; fbU::AbstractArray{ComplexF64,3}, Jw::Vector{T}, β::Real, dt::Real, ntimes::Int, kmax::Union{Int,Nothing}=nothing, extraargs::QuAPIArgs=QuAPIArgs(), svec=[1.0 -1.0], reference_prop=false, verbose::Bool=false, output::Union{Nothing,HDF5.Group}=nothing, from_TTM::Bool=false, exec=ThreadedEx()) where {T<:SpectralDensities.SpectralDensity}
    @assert length(Jw) == size(svec, 1)
    nb = length(Jw)
    η = [EtaCoefficients.calculate_η(jw; β, dt, kmax=ntimes, imaginary_only=reference_prop) for jw in Jw]
    sdim = size(fbU, 2)
    sdim2 = sdim^2
    state_values, _ = setup_simulation(ones(sdim, sdim), η, svec, extraargs)

    if verbose
        @info "Starting propagation within memory"
    end
    U0e = Array{ComplexF64}(undef, ntimes, sdim2, sdim2)
    T0e = zero(U0e)
    if !isnothing(output)
        if !from_TTM
            Utilities.check_or_insert_value(output, "U0e", U0e)
        end
        Utilities.check_or_insert_value(output, "T0e", T0e)
        Utilities.check_or_insert_value(output, "time_taken", zeros(Float64, ntimes))
        Utilities.check_or_insert_value(output, "num_paths", zeros(Int64, ntimes))
    end
    nkinks = extraargs.num_kinks==-1 ? ntimes : extraargs.num_kinks
    nblips = extraargs.num_blips==-1 ? ntimes+1 : extraargs.num_blips
    forward_paths = [[j] for j = UInt8(1):UInt8(sdim)]
    forward_amplitudes = [1.0+0.0im for j = 1:sdim]
    for i = 1:ntimes
        if verbose
            @info "Starting time step $(i)"
        end
        _, time_taken, memory_allocated, gc_time, _ = @timed begin
            forward_paths, forward_amplitudes = Utilities.generate_paths_kink_limit(forward_paths, forward_amplitudes, nkinks, sdim, fbU, extraargs.prop_cutoff, sqrt(extraargs.cutoff))
            num_paths = length(forward_paths)
            if verbose
                @info "Path generation done. $(num_paths) forward paths generated. Expect up to $(num_paths^2) forward-backward paths based on filtration."
            end
            # @floop exec for ((fp, fa), (bp, ba)) in Iterators.product(zip(forward_paths, forward_amplitudes), zip(forward_paths, forward_amplitudes))
            let forward_paths = forward_paths, forward_amplitudes = forward_amplitudes
                @floop exec for j = 1:num_paths, k = 1:num_paths
                    @inbounds bare_amplitude = forward_amplitudes[j] * conj(forward_amplitudes[k])
                    if abs(bare_amplitude) < extraargs.cutoff
                        continue
                    end
                    @init fp = zeros(UInt16, i+1)
                    @init bp = zeros(UInt16, i+1)
                    @init states = zeros(UInt16, i+1)
                    num_blips = 0
                    for l = 1:i+1
                        fp[l] = forward_paths[j][l]
                        bp[l] = forward_paths[k][l]
                        states[l] = (fp[l] - 1) * sdim + bp[l]
                        if fp[l] != bp[l]
                            num_blips += 1
                        end
                    end
                    if num_blips > nblips
                        continue
                    end
                    @reduce num_paths = 0 + 1
                    for bn = 1:nb
                        @inbounds bare_amplitude *= get_path_influence(η[bn], bn, state_values, states, false)
                    end
                    @init tmpval = zeros(ComplexF64, sdim2, sdim2)
                    fill!(tmpval, 0.0+0.0im)
                    @inbounds tmpval[states[end], states[1]] = bare_amplitude
                    @reduce tmpU0e = zeros(ComplexF64, sdim2, sdim2) .+ tmpval
                end
                @inbounds U0e[i, :, :] .= tmpU0e
            end
            TTM.update_Ts!(T0e, U0e, i)
        end
        if !isnothing(output)
            output["U0e"][i, :, :] = U0e[i, :, :]
            output["T0e"][i, :, :] = T0e[i, :, :]
            output["time_taken"][i] = time_taken
            output["num_paths"][i] = num_paths
            flush(output)
        end
        if verbose
            @info "Done time step $(i); # paths = $(sum(num_paths)); norm(Tmat) = $(round(norm(T0e[i, :, :]); digits=3)); time = $(round(time_taken; digits=3)) sec; memory allocated = $(round(memory_allocated / 1024^3; digits=3)) GiB; gc time = $(round(gc_time; digits=3)) sec"
        end
    end
    U0e
end

function propagate_kink(; fbU::AbstractArray{ComplexF64,3}, Jw::Vector{T}, ρ0::AbstractMatrix{ComplexF64}, β::Real, dt::Real, ntimes::Int, extraargs::QuAPIArgs=QuAPIArgs(), svec=[1.0 -1.0], reference_prop=false, verbose::Bool=false, output::Union{Nothing,HDF5.Group}=nothing, from_TTM::Bool=false, exec=ThreadedEx()) where {T<:SpectralDensities.SpectralDensity}
    @assert length(Jw) == size(svec, 1)
    η = [EtaCoefficients.calculate_η(jw; β, dt, kmax=ntimes, imaginary_only=reference_prop) for jw in Jw]
    sdim = size(fbU, 2)
    sdim2 = sdim^2
    state_values, _ = setup_simulation(ones(sdim, sdim), η, svec, extraargs)

    if verbose
        @info "Starting propagation within memory"
    end
    ρs = zeros(ComplexF64, ntimes+1, sdim, sdim)
    ρs[1, :, :] = ρ0
    if !isnothing(output)
        Utilities.check_or_insert_value(output, "rho", ρs)
        Utilities.check_or_insert_value(output, "time_taken", zeros(Float64, ntimes))
        Utilities.check_or_insert_value(output, "num_paths", zeros(Int64, ntimes))
    end
    nkinks = extraargs.num_kinks==-1 ? ntimes : extraargs.num_kinks
    nblips = extraargs.num_blips==-1 ? ntimes+1 : extraargs.num_blips
    numpaths = zeros(Int64, ntimes)
    timetaken = zeros(Float64, ntimes)
    for ind in findall(!iszero, ρ0)
        forward_paths = [[UInt8(ind[1])]]
        forward_amplitudes = [1.0+0.0im]
        backward_paths = [[UInt8(ind[2])]]
        backward_amplitudes = [1.0+0.0im]
        for i = 1:ntimes
            if verbose
                @info "Starting time step $(i)"
            end
            _, time_taken, memory_allocated, gc_time, _ = @timed begin
                forward_paths, forward_amplitudes = Utilities.generate_paths_kink_limit(forward_paths, forward_amplitudes, nkinks, sdim, fbU, extraargs.prop_cutoff, sqrt(extraargs.cutoff))
                backward_paths, backward_amplitudes = Utilities.generate_paths_kink_limit(backward_paths, backward_amplitudes, nkinks, sdim, fbU, extraargs.prop_cutoff, sqrt(extraargs.cutoff))
                if verbose
                    @info "Path generation done. $(length(forward_paths)) forward paths generated. Expect up to $(length(forward_paths)^2) forward-backward paths based on filtration."
                end
                @floop exec for ((fp, fa), (bp, ba)) in Iterators.product(zip(forward_paths, forward_amplitudes), zip(backward_paths, backward_amplitudes))
                    num_blips = count(fp .!= bp)
                    if num_blips > nblips
                        continue
                    end
                    bare_amplitude = fa * conj(ba)
                    @init states = Vector{UInt16}(undef, i+1)
                    @inbounds for j in eachindex(fp)
                        states[j] = (fp[j] - 1) * sdim + bp[j]
                    end
                    @reduce num_paths = 0 + 1
                    for (bn, bη) in enumerate(η)
                        @inbounds bare_amplitude *= get_path_influence(bη, bn, state_values, states, false)
                    end
                    @init tmprho = zeros(ComplexF64, sdim2)
                    tmprho .= 0.0
                    @inbounds tmprho[states[end]] = bare_amplitude * ρ0[ind]
                    @reduce tmprhot = zeros(ComplexF64, sdim2) .+ tmprho
                end
                numpaths[i] += num_paths
                @inbounds ρs[i+1, :, :] .+= Utilities.density_matrix_vector_to_matrix(tmprhot)
            end
            timetaken[i] += time_taken
            if verbose
                @info "Done time step $(i); # paths = $(sum(num_paths)); time = $(round(time_taken; digits=3)) sec; memory allocated = $(round(memory_allocated / 1024^3; digits=3)) GiB; gc time = $(round(gc_time; digits=3)) sec"
            end
            if !isnothing(output)
                output["rho"][i+1, :, :] = ρs[i+1, :, :]
                output["time_taken"][i] = timetaken[i]
                output["num_paths"][i] = numpaths[i]
                flush(output)
            end
        end
    end
    0:dt:ntimes*dt, ρs
end
end
