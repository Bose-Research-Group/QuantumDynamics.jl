module ComplexQuAPI

using FLoops
using HDF5
using LinearAlgebra
using ..BMatrix, ..ComplexPISetup, ....SpectralDensities, ....QuAPI, ......Utilities

const references = """
- Topaler, M.; Makri, N. Quantum rates for a double well coupled to a dissipative bath: Accurate path integral results and comparison with approximate theories. The Journal of Chemical Physics 1994, 101 (9), 7500-7519.
- Bose, A. Quantum correlation functions through tensor network path integral. The Journal of Chemical Physics 2023, 159 (21), 214110."""

"""
    A_of_t(; Hamiltonian::AbstractMatrix{ComplexF64}, β::Float64, t::Float64, N::Int64, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, svec::AbstractMatrix{Float64}, A, extraargs::QuAPI.QuAPIArgs=QuAPI.QuAPIArgs(), exec=FLoops.ThreadedEx())
Calculates ``Tr_{env}(U(t) exp(-β H/2) A exp(-β H/2) U^{-1}(t))`` for a system interacting with an environment at a time-point `t` using the tensor network path integral method. This can be used for thermodynamics or for calculating correlation functions.

Arguments:
- `Hamiltonian`: system Hamiltonian
- `Jw`: array of spectral densities
- `svec`: diagonal elements of system operators through which the corresponding baths interact. QuAPI currently only works for baths with diagonal coupling to the system.
- `β`: inverse temperature
- `t`: time at which the function is evaluated
- `N`: number of path integral discretizations
- `A`: system operator to be evaluated
- `extraargs`: extra arguments for the tensor network algorithm. Contains the `cutoff` threshold for SVD filtration, the maximum bond dimension, `maxdim`, and the `algorithm` of applying an MPO to an MPS.
- `exec`: FLoops.jl execution policy
"""
function A_of_t(; Hamiltonian::AbstractMatrix{ComplexF64}, β::Float64, t::Float64, N::Int64, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, svec::AbstractMatrix{Float64}, A, extraargs::QuAPI.QuAPIArgs=QuAPI.QuAPIArgs(), exec=FLoops.ThreadedEx(), type_corr="symm")
    @assert length(Jw) == size(svec, 1)
    nbaths = length(Jw)
    ((U, Udag), tarr) = if type_corr == "symm"
        (ComplexPISetup.get_complex_time_propagator(Hamiltonian, β, t, N), ComplexPISetup.get_complex_time_array(t, β, N))
    elseif type_corr == "asymm"
        (ComplexPISetup.get_asymm_time_propagator(Hamiltonian, β, t, N), ComplexPISetup.get_asymm_time_array(t, β, N))
    else
        error("Unknown type_corr = $(type_corr); expected symm or asymm")
    end
    npoints = 2N+2
    Bmat = [BMatrix.get_B_matrix(J, β, N, tarr) for J in Jw]
    nfor = npoints ÷ 2
    sdim = size(Hamiltonian, 1)
    num_paths = sdim^npoints
    @inbounds begin
        @floop exec for path_num = 1:num_paths
            @init states = Vector{Int}(undef, npoints)
            # @init states = Utilities.unhash_path(path_num, npoints - 1, sdim)
            Utilities.unhash_path!(states, path_num, npoints-1, sdim)
            # e^{-i H tc} A e^{i H tc}
            val = A[states[nfor+1], states[nfor]]
            if abs(val) ≤ extraargs.cutoff
                continue
            end
            for j = 1:nfor-1
                val *= Udag[states[j+1], states[j]]
            end
            for j = nfor+1:npoints-1
                val *= U[states[j+1], states[j]]
            end
            if abs(val) ≤ extraargs.cutoff
                continue
            end
            infl = 0.0 + 0.0im
            for nb = 1:nbaths
                nnonzeros = 0
                for s in states
                    if svec[nb, s] != 0
                        nnonzeros += 1
                    end
                end
                ks = zeros(Int64, nnonzeros)
                svecs = zeros(nnonzeros)
                l = 1
                for (k, s) in enumerate(states)
                    if svec[nb, s] != 0
                        svecs[l] = svec[nb, s]
                        ks[l] = k
                        l += 1
                    end
                end
                for (i, k) in enumerate(ks)
                    for kp = 1:i
                        infl -= Bmat[nb][k, ks[kp]] * svecs[i] * svecs[kp]
                    end
                end
            end
            @reduce num_paths_local = 0 + 1
            @init tmpA = zeros(ComplexF64, sdim, sdim)
            tmpA .= 0
            @inbounds tmpA[states[end], states[1]] = val * exp(infl)
            @reduce At = zeros(ComplexF64, sdim, sdim) .+ tmpA
        end
    end
    At, num_paths_local
end

"""
    complex_correlation_function(; Hamiltonian::AbstractMatrix{ComplexF64}, β::Real, tfinal::Real, dt::Real, N::Int, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, svec::AbstractMatrix{<:Real}, A, B, Z::Real, extraargs::QuAPI.QuAPIArgs=QuAPI.QuAPIArgs(), verbose::Bool=false, output::Union{Nothing,HDF5.Group}=nothing, exec=ThreadedEx())
Calculates the ``<B(t_c) A(0)> / Z`` correlation function for a system interacting with an environment upto a maximum time of `tfinal` with a time-step of `dt` using the tensor network path integral method.

Arguments:
- `Hamiltonian`: system Hamiltonian
- `Jw`: array of spectral densities
- `svec`: diagonal elements of system operators through which the corresponding baths interact. QuAPI currently only works for baths with diagonal coupling to the system.
- `β`: inverse temperature
- `tfinal`: maximum time till which the correlation function is evaluated
- `dt`: time-step for evaluating correlation function
- `N`: number of path integral discretizations
- `A`: system operator evaluated at time zero
- `B`: system operator evaluated at time `t`
- `Z`: partition function for normalization
- `extraargs`: extra arguments for the tensor network algorithm. Contains the `cutoff` threshold for SVD filtration, the maximum bond dimension, `maxdim`, and the `algorithm` of applying an MPO to an MPS.
- `verbose`: verbosity
- `output`: output HDF5 file for storage of results
- `exec`: FLoops.jl execution policy
"""
function complex_correlation_function(; Hamiltonian::AbstractMatrix{ComplexF64}, β::Real, tfinal::Real, dt::Real, N::Int, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, svec::AbstractMatrix{<:Real}, A, B, Z::Real, extraargs::QuAPI.QuAPIArgs=QuAPI.QuAPIArgs(), verbose::Bool=false, output::Union{Nothing,HDF5.Group}=nothing, exec=ThreadedEx(), type_corr="symm")
    time = 0:dt:tfinal |> collect
    nbaths = length(Jw)
    corr = zeros(ComplexF64, length(time), length(B))
    num_paths = zeros(Int64, length(time))
    if !isnothing(output)
        Utilities.check_or_insert_value(output, "time_taken", zeros(Float64, length(time)))
        Utilities.check_or_insert_value(output, "corr", corr)
        Utilities.check_or_insert_value(output, "num_paths", num_paths)
    end
    for (i, t) in enumerate(time)
        _, time_taken, memory_allocated, gc_time, _ = @timed begin
            At, np = A_of_t(; Hamiltonian, β, t, N, Jw, svec, A, extraargs, exec, type_corr)
        end
        At /= Z
        corr[i, :] .= [tr(b * At) for b in B]
        num_paths[i] = np
        if verbose
            @info "Step = $(i); number of paths = $(np), time = $(round(time_taken; digits=3)) sec; memory allocated = $(round(memory_allocated / 1e6; digits=3)) GB; gc time = $(round(gc_time; digits=3)) sec"
        end
        if !isnothing(output)
            output["corr"][i, :] = corr[i, :]
            output["time_taken"][i] = time_taken
            output["num_paths"][i] = np
            flush(output)
        end
    end
    time, corr, num_paths
end

"""
    adaptive_kink_A_of_t(; Hamiltonian::AbstractMatrix{ComplexF64}, β::Float64, t::Float64, N::Int64, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, svec::AbstractMatrix{Float64}, A, extraargs::QuAPI.QuAPIArgs=QuAPI.QuAPIArgs(), exec=FLoops.ThreadedEx())
Calculates ``Tr_{env}(U(t) exp(-β H/2) A exp(-β H/2) U^{-1}(t))`` for a system interacting with an environment at a time-point `t` using the tensor network path integral method. This can be used for thermodynamics or for calculating correlation functions.

Arguments:
- `Hamiltonian`: system Hamiltonian
- `Jw`: array of spectral densities
- `svec`: diagonal elements of system operators through which the corresponding baths interact. QuAPI currently only works for baths with diagonal coupling to the system.
- `β`: inverse temperature
- `t`: time at which the function is evaluated
- `N`: number of path integral discretizations
- `A`: system operator to be evaluated
- `extraargs`: extra arguments for the tensor network algorithm. Contains the `cutoff` threshold for SVD filtration, the maximum bond dimension, `maxdim`, and the `algorithm` of applying an MPO to an MPS.
- `exec`: FLoops.jl execution policy
"""
function adaptive_kink_A_of_t(; Hamiltonian::AbstractMatrix{ComplexF64}, β::Float64, t::Float64, N::Int64, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, svec::AbstractMatrix{Float64}, A, extraargs::QuAPI.QuAPIArgs=QuAPI.QuAPIArgs(), exec=FLoops.ThreadedEx(), type_corr="symm")
    @assert length(Jw) == size(svec, 1)
    sdim = size(Hamiltonian, 1)
    nbaths = length(Jw)
    ((U, Udag), tarr) = if type_corr == "symm"
        (ComplexPISetup.get_complex_time_propagator(Hamiltonian, β, t, N), ComplexPISetup.get_complex_time_array(t, β, N))
    elseif type_corr == "asymm"
        (ComplexPISetup.get_asymm_time_propagator(Hamiltonian, β, t, N), ComplexPISetup.get_asymm_time_array(t, β, N))
    else
        error("Unknown type_corr = $(type_corr); expected symm or asymm")
    end
    Us = zeros(ComplexF64, N, sdim, sdim)
    Udags = zeros(ComplexF64, N, sdim, sdim)
    for j = 1:N
        Us[j, :, :] .= U
        Udags[j, :, :] .= Udag
    end
    npoints = 2N+2
    Bmat = [BMatrix.get_B_matrix(J, β, N, tarr) for J in Jw]
    nkinks = extraargs.num_kinks==-1 ? N : extraargs.num_kinks
    Attotal = zero(A)
    total_num_paths = 0
    @inbounds begin
        for ind in findall(!iszero, A)
            forward_paths, forward_amplitudes = Utilities.generate_paths_kink_limit(UInt8(ind[1]), N+1, nkinks, sdim, Us, extraargs.prop_cutoff, sqrt(extraargs.cutoff))
            backward_paths, backward_amplitudes = Utilities.generate_paths_kink_limit(UInt8(ind[2]), N+1, nkinks, sdim, Udags, extraargs.prop_cutoff, sqrt(extraargs.cutoff), false)
            @floop exec for ((fp, fa), (bp, ba)) in Iterators.product(zip(forward_paths, forward_amplitudes), zip(backward_paths, backward_amplitudes))
                val = fa * ba * A[fp[1], bp[1]]
                if abs(val) ≤ extraargs.cutoff
                    continue
                end

                @init states = zeros(Int64, npoints)
                @init ks     = zeros(Int64, npoints)
                @init svecs  = zeros(Float64, npoints)

                idx = 1
                @inbounds for k in length(bp):-1:1
                    states[idx] = bp[k]
                    idx += 1
                end
                @inbounds for k in 1:length(fp)
                    states[idx] = fp[k]
                    idx += 1
                end

                infl = 0.0 + 0.0im
                for nb = 1:nbaths
                    nnonzeros = 0
                    for k in 1:npoints
                        s = states[k]
                        sv = svec[nb, s]
                        if sv != 0
                            nnonzeros += 1
                            ks[nnonzeros] = k
                            svecs[nnonzeros] = sv
                        end
                    end
                    for i in 1:nnonzeros
                        k = ks[i]
                        for kp = 1:i
                            infl -= Bmat[nb][k, ks[kp]] * svecs[i] * svecs[kp]
                        end
                    end
                end
                @reduce num_paths = 0 + 1
                @init tmpA = zeros(ComplexF64, sdim, sdim)
                tmpA .= 0
                @inbounds tmpA[states[end], states[1]] = val * exp(infl)
                @reduce At = zeros(ComplexF64, sdim, sdim) .+ tmpA
            end
            Attotal += At
            total_num_paths += num_paths
        end
    end
    Attotal, total_num_paths
end

"""
    adaptive_kink_complex_correlation_function(; Hamiltonian::AbstractMatrix{ComplexF64}, β::Real, tfinal::Real, dt::Real, N::Int, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, svec::AbstractMatrix{<:Real}, A, B, Z::Real, extraargs::QuAPI.QuAPIArgs=QuAPI.QuAPIArgs(), verbose::Bool=false, output::Union{Nothing,HDF5.Group}=nothing, exec=ThreadedEx())
Calculates the ``<B(t_c) A(0)> / Z`` correlation function for a system interacting with an environment upto a maximum time of `tfinal` with a time-step of `dt` using the tensor network path integral method.

Arguments:
- `Hamiltonian`: system Hamiltonian
- `Jw`: array of spectral densities
- `svec`: diagonal elements of system operators through which the corresponding baths interact. QuAPI currently only works for baths with diagonal coupling to the system.
- `β`: inverse temperature
- `tfinal`: maximum time till which the correlation function is evaluated
- `dt`: time-step for evaluating correlation function
- `N`: number of path integral discretizations
- `A`: system operator evaluated at time zero
- `B`: system operator evaluated at time `t`
- `Z`: partition function for normalization
- `extraargs`: extra arguments for the tensor network algorithm. Contains the `cutoff` threshold for SVD filtration, the maximum bond dimension, `maxdim`, and the `algorithm` of applying an MPO to an MPS.
- `verbose`: verbosity
- `output`: output HDF5 file for storage of results
- `exec`: FLoops.jl execution policy
"""
function adaptive_kink_complex_correlation_function(; Hamiltonian::AbstractMatrix{ComplexF64}, β::Real, tfinal::Real, dt::Real, N::Int, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, svec::AbstractMatrix{<:Real}, A, B, Z::Real, extraargs::QuAPI.QuAPIArgs=QuAPI.QuAPIArgs(), verbose::Bool=false, output::Union{Nothing,HDF5.Group}=nothing, exec=ThreadedEx(), type_corr="symm")
    time = 0:dt:tfinal |> collect
    nbaths = length(Jw)
    corr = zeros(ComplexF64, length(time), length(B))
    num_paths = zeros(Int64, length(time))
    if !isnothing(output)
        Utilities.check_or_insert_value(output, "time_taken", zeros(Float64, length(time)))
        Utilities.check_or_insert_value(output, "corr", corr)
        Utilities.check_or_insert_value(output, "num_paths", num_paths)
    end
    for (i, t) in enumerate(time)
        _, time_taken, memory_allocated, gc_time, _ = @timed begin
            At, np = adaptive_kink_A_of_t(; Hamiltonian, β, t, N, Jw, svec, A, extraargs, exec, type_corr)
        end
        At /= Z
        corr[i, :] .= [tr(b * At) for b in B]
        num_paths[i] = np
        if verbose
            @info "Step = $(i); number of paths = $(np), time = $(round(time_taken; digits=3)) sec; memory allocated = $(round(memory_allocated / 1e6; digits=3)) GB; gc time = $(round(gc_time; digits=3)) sec"
        end
        if !isnothing(output)
            output["corr"][i, :] = corr[i, :]
            output["time_taken"][i] = time_taken
            output["num_paths"][i] = np
            flush(output)
        end
    end
    time, corr, num_paths
end

end
