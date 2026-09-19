module PLDM

using HDF5
using ..Utilities
using ..Solvents, ..Systems, ..SpectralDensities

const references = """
- Huo, P.; Coker, D. F. Communication: Partial linearized density matrix dynamics for dissipative, non-adiabatic quantum evolution. J. Chem. Phys. 2011, 135, 201101.
- Bonella, S.; Coker, D. F. LAND-map, a linearised approach to nonadiabatic dynamics using the mapping formalism. J. Chem. Phys. 2005, 122, 194102.
- Huo, P.; Coker, D. F. Consistent schems for non-adiabatic dynamics derived from partial linearized density matrix propagation. J. Chem. Phys. 2012, 137, 22A535."""

struct PLDMSysPhaseSpace <: Systems.PartialLinearisedSysPhaseSpace
    Xf::AbstractVector{<:Real}
    Pf::AbstractVector{<:Real}
    Xb::AbstractVector{<:Real}
    Pb::AbstractVector{<:Real}
end

struct PLDMSys <: Systems.MappedSystem
    h::AbstractMatrix{<:Complex}
    ρ₀::AbstractMatrix{<:Complex}
    d::Integer
    bath::Solvents.Solvent
    nsamples::Integer
end
function PLDMSys(; Hamiltonian::AbstractMatrix{<:Complex},
                 ρ₀::AbstractMatrix{<:Complex},
                 bath::Solvents.Solvent, nsamples::Integer)
    @assert nsamples == length(bath)
    d = size(Hamiltonian, 1)
    PLDMSys(Hamiltonian, ρ₀, d, bath, nsamples)
end

function Base.iterate(sys::PLDMSys, state=1)
    state > sys.nsamples && return nothing

    bathps, _ = iterate(sys.bath, state)
    Xf, Pf = randn(sys.d), randn(sys.d)
    Xb, Pb = randn(sys.d), randn(sys.d)

    (PLDMSysPhaseSpace(Xf, Pf, Xb, Pb), bathps), state+1
end
Base.eltype(::PLDMSys) = PLDMSysPhaseSpace
Base.length(s::PLDMSys) = s.nsamples
Base.firstindex(::PLDMSys) = 1
Base.getindex(s::PLDMSys, n::Integer) = iterate(s, n)[1]
Systems.γ(::PLDMSys) = 0.0

function Systems.transform_op(sys::PLDMSys, op::Union{AbstractMatrix,AbstractVector},
                              ps::PLDMSysPhaseSpace, path::Symbol)
    @assert path ∈ [ :forward, :backward ]
    if path == :forward
        Systems.transform_op(sys, op, ps.Xf, ps.Pf)
    else
        Systems.transform_op(sys, op, ps.Xb, ps.Pb)
    end
end

function build_ρ!(sys::PLDMSys, sps0::PLDMSysPhaseSpace,
                  Xf::AbstractVector{<:Real},
                  Pf::AbstractVector{<:Real},
                  Xb::AbstractVector{<:Real},
                  Pb::AbstractVector{<:Real},
                  ρ::AbstractMatrix{<:Complex})
    d = sys.d
    zf0 = sps0.Xf + im * sps0.Pf
    zb0 = sps0.Xb + im * sps0.Pb
    w = zf0' * sys.ρ₀ * zb0 / 2
    w′ = zb0' * sys.ρ₀ * zf0 / 2

    ρ .= ((w  * (Xf + im * Pf) * (Xb + im * Pb)') / 2 +
          (w′ * (Xb + im * Pb) * (Xf + im * Pf)') / 2) / 2
end

function propagate_trajectory(sys::PLDMSys, sps0::PLDMSysPhaseSpace,
                              bps0::Solvents.PhaseSpace,
                              dt::Real, ntimes::Integer)
    Xf = similar(sps0.Xf)
    Xf .= sps0.Xf
    Pf = similar(sps0.Pf)
    Pf .= sps0.Pf
    Xb = similar(sps0.Xb)
    Xb .= sps0.Xb
    Pb = similar(sps0.Pb)
    Pb .= sps0.Pb
    buf = similar(Pb)
    bps₀ = bps0
    bpsₙ = typeof(bps0)(similar.(bps0.q), similar.(bps0.p))
    d = sys.d

    ρ = zeros(ComplexF64, ntimes+1,d,d)

    @views build_ρ!(sys, sps0, Xf, Pf, Xb, Pb, ρ[1,:,:])

    dt2 = dt / 2
    bs = sys.bath
    A = zeros(d,d)
    s̄c = similar.(bs.c)
    Systems.Fbath!(sys, sps0, s̄c)
    sps = PLDMSysPhaseSpace(Xf, Pf, Xb, Pb)
    @inbounds for t in 2:ntimes+1
        Solvents.propagate_forced_bath!(bs, bps₀, bpsₙ, s̄c, dt2, 1)

        sinA, cosA = Systems.get_propagator(sys, bpsₙ, A, dt)
        Systems.apply_propagator!(sys, sps, sinA, cosA, buf)

        Systems.Fbath!(sys, sps, s̄c)
        Solvents.propagate_forced_bath!(bs, bpsₙ, bps₀, s̄c, dt2, 1)

        @views build_ρ!(sys, sps0, Xf, Pf, Xb, Pb, ρ[t,:,:])
    end

    ρ
end

function propagate_trajectories(sys::PLDMSys, dt::Real, ntimes::Integer;
                                output::Union{Nothing,HDF5.Group}=nothing,
                                verbose::Bool=false, kwargs...)
    ρ = zeros(ComplexF64, ntimes+1,sys.d,sys.d)
    outputρ = if haskey(kwargs, :outgroup)
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
            ndone += 1
            ρ += ρᵢ
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
    propagate(; Hamiltonian::Matrix{<:Complex}, Jw::Vector{T},
              β::Real, num_osc::Vector{<:Integer}, svec::Matrix{<:Real},
              ρ0::Matrix{<:Complex}, dt::Real,
              ntimes::Integer, nmc::Integer, verbose::Bool=false,
              kwargs...) where {T<:SpectralDensities.SpectralDensity}

Propagate the system using the spin-mapped PLDM method.

Arguments:
- `ρ0`: initial reduced density matrix
- `Hamiltonian`: the Hamiltonian of the sub-system
- `Jw`: list of spectral densities
- `β`: the inverse temperature of the bath
- `num_osc`: the number of oscillators for each bath
- `svec`: diagonal elements of system operators through which the
  corresponding baths interact
- `dt`: the time step for the propagation
- `nmc`: the number of Monte-Carlo samples

Propagate the reduced density matrix `ρ0` using the partial linearized
density matrix propagation scheme.
"""
function propagate(; Hamiltonian::Matrix{<:Complex}, Jw::Vector{T},
                   β::Real, num_osc::Vector{<:Integer}, svec::Matrix{<:Real},
                   ρ0::Matrix{<:Complex}, dt::Real,
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
    sys = PLDMSys(; Hamiltonian, ρ₀=ρ0, bath, nsamples=nmc)

    propagate_trajectories(sys, dt, ntimes; verbose, kwargs...)
end

end
