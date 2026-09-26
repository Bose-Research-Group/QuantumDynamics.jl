module LSC

using HDF5
using ..Utilities
using ..Solvents, ..Systems, ..SpectralDensities
using LinearAlgebra: Diagonal, tr
using LinearAlgebra: I as Id
using Distributions: MvNormal

const references = """
- Sun, X.; Miller, W. H. Mixed semiclassical–classical approaches to the dynamics of complex molecular systems. J. Chem. Phys. 1997, 106, 916–927.
- Sun, X.; Wang, H.; Miller, W. H. Semiclassical theory of electronically nonadiabatic dynamics: Results of a linearized approximation to the initial value representation. J. Chem. Phys. 1998, 109, 7064–7074."""

struct LSCSysPhaseSpace <: Systems.LinearisedSysPhaseSpace
    X::AbstractVector{<:Real}
    P::AbstractVector{<:Real}
end

struct LSCSys <: Systems.MappedSystem
    h::AbstractMatrix{<:Complex}
    ρ₀::AbstractMatrix{<:Complex}
    d::Integer
    bath::Solvents.Solvent
    nsamples::Integer
    XPdist::MvNormal
end
function LSCSys(; Hamiltonian::AbstractMatrix{<:Complex},
                ρ₀::AbstractMatrix{<:Complex},
                bath::Solvents.Solvent, nsamples::Integer)
    @assert nsamples == length(bath)
    d = size(Hamiltonian, 1)
    μ = zeros(d)
    Σ⁻¹ = Diagonal(fill(4.0, d))
    dist = MvNormal(μ, inv(Σ⁻¹))

    LSCSys(Hamiltonian, ρ₀, d, bath, nsamples, dist)
end

function Base.iterate(sys::LSCSys, state=1)
    state > sys.nsamples && return nothing

    bathps, _ = iterate(sys.bath, state)
    X, P = rand(sys.XPdist), rand(sys.XPdist)

    ((LSCSysPhaseSpace(X, P)), bathps), state+1
end
Base.eltype(::LSCSys) = LSCSysPhaseSpace
Base.length(s::LSCSys) = s.nsamples
Base.firstindex(::LSCSys) = 1
Base.getindex(s::LSCSys, n::Integer) = iterate(s, n)[1]
Systems.γ(s::LSCSys) = 1.0

Systems.transform_op(sys::LSCSys, op::Union{AbstractVector,AbstractMatrix}, ps::LSCSysPhaseSpace) =
    Systems.transform_op(sys, op, ps.X, ps.P)

sampling_weight(::LSCSys, ρ₀::AbstractMatrix{<:Complex}, ps::LSCSysPhaseSpace) =
    4 * (ps.X' * ρ₀ * ps.X + ps.P' * ρ₀ * ps.P + im * (ps.X' * ρ₀ * ps.P - ps.P' * ρ₀ * ps.X) - tr(ρ₀) / 2)

function build_ρ!(sys::LSCSys, X::AbstractVector{<:Real}, P::AbstractVector{<:Real}, ρ::AbstractMatrix{<:Complex})
    ρ .= (X * X' + P * P' + im * (P * X' - X * P') - 0.5Id(sys.d))
end

function propagate_trajectory(sys::LSCSys, sps0::LSCSysPhaseSpace,
                              bps0::Solvents.PhaseSpace, dt::Real, ntimes::Integer)
    X = sps0.X
    P = sps0.P
    buf = similar(sps0.X)
    bps₀ = bps0
    bpsₙ = typeof(bps0)(similar.(bps0.q), similar.(bps0.p))
    d = sys.d

    ρ = zeros(ComplexF64, ntimes+1,d,d)
    w₀ = sampling_weight(sys, sys.ρ₀, sps0)
    @views build_ρ!(sys, X, P, ρ[1,:,:])

    dt2 = dt/2
    bs = sys.bath
    A = zeros(d,d)
    sc = similar.(bs.c)
    Systems.Fbath!(sys, sps0, sc)
    sps = LSCSysPhaseSpace(X, P)
    @inbounds for t in 2:ntimes+1
        Solvents.propagate_forced_bath!(bs, bps₀, bpsₙ, sc, dt2, 1)

        sinA, cosA = Systems.get_propagator(sys, bpsₙ, A, dt)
        Systems.apply_propagator!(sys, sps, sinA, cosA, buf)

        Systems.Fbath!(sys, sps, sc)
        Solvents.propagate_forced_bath!(bs, bpsₙ, bps₀, sc, dt2, 1)

        @views build_ρ!(sys, X, P, ρ[t,:,:])
    end
    ρ .*= w₀

    ρ
end

function propagate_trajectories(sys::LSCSys, dt::Real, ntimes::Integer;
                                output::Union{Nothing,HDF5.Group}=nothing, verbose::Bool=false,
                                kwargs...)
    ρ = zeros(ComplexF64, ntimes+1,sys.d,sys.d)

    outputρ = if !isnothing(output) && haskey(kwargs, :outgroup)
        Utilities.create_and_select_group(output, kwargs[:outgroup])
    else
        nothing
    end

    mutlock = ReentrantLock()
    ndone = 0
    nthreads = Threads.nthreads()
    stats = @timed Threads.@threads for (sps0, bps0) in sys
        ρᵢ = propagate_trajectory(sys, sps0, bps0, dt, ntimes)
        lock(mutlock) do
            ρ .+= ρᵢ
            ndone += 1
            verbose && ndone % nthreads == 0 &&
                @info "Trajectories complete: $(100ndone / length(sys))%"
        end
    end
    @info "All trajectories complete"
    @info "Time taken = $(round(stats.time; digits=3)) sec; memory allocated = $(round(stats.bytes / 1e9; digits=3)) GB; gc time = $(round(stats.gctime; digits=3)) sec"

    ρ ./= length(sys)
    if !isnothing(outputρ)
        outputρ["rho"] = ρ
        flush(outputρ)
    end

    ρ
end

"""
    propagate(; Hamiltonian::AbstractMatrix{<:Complex}, Jw::Vector{T},
              β::Real, num_osc::Vector{<:Integer}, svec::Matrix{<:Real},
              ρ0::AbstractMatrix{<:Complex}, dt::Real,
              ntimes::Integer, nmc::Integer, verbose::Bool=false,
              kwargs...) where {T<:SpectralDensities.SpectralDensity}

Propagate the system via linearised semiclassics after MMST mapping.

Arguments:
- `ρ0`: initial reduced density matrix
- `Hamiltonian`: the Hamiltonian of the sub-system
- `Jw0`: list of spectral densities
- `β`: the inverse temperature of the bath
- `num_osc`: the number of oscillators for each bath
- `svec`: diagonal elements of system operators through which the
  corresponding bath interact
- `dt`: the time step for the propagation
- `nmc`: the number of Monte-Carlo samples

Propagate the reduced density matrix `ρ0` by doing fully linearised
semiclassics on the system space, after transforming the Hamiltonian
with the Meyer-Miller-Stock-Thoss mapping procedure.
"""
function propagate(; Hamiltonian::AbstractMatrix{<:Complex}, Jw::Vector{T},
                   β::Real, num_osc::Vector{<:Integer}, svec::Matrix{<:Real},
                   ρ0::AbstractMatrix{<:Complex}, dt::Real,
                   ntimes::Integer, nmc::Integer, verbose::Bool=false,
                   kwargs...) where {T<:SpectralDensities.SpectralDensity}
    nbaths = length(Jw)
    c = Vector{Vector{Float64}}(undef, nbaths)
    ω = Vector{Vector{Float64}}(undef, nbaths)
    s = Vector{Vector{Float64}}(undef, nbaths)

    for n in 1:nbaths
        ω[n], c[n] = SpectralDensities.discretize(Jw[n], num_osc[n])
        s[n] = svec[n,:]
    end

    bath = Solvents.HarmonicBath(; β, ω, c, svecs=s, nsamples=nmc)
    sys = LSCSys(; Hamiltonian, ρ₀=ρ0, bath, nsamples=nmc)

    propagate_trajectories(sys, dt, ntimes; verbose, kwargs...)
end

end
