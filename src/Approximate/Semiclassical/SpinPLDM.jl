module SpinPLDM

using HDF5
using ..Utilities
using ..Solvents, ..Systems, ..SpectralDensities
using LinearAlgebra: diagm

const references = """
- Mannouch, J. R.; Richardson, J. O. A partially linearised spin-mapping approach for non-adiabatic dynamics. I. Derivation of the theory. J. Chem. Phys. 2020 153, 194109."""

struct SpinPLDMSysPhaseSpace <: Systems.PartialLinearisedSysPhaseSpace
    Xf::AbstractVector{<:Real}
    Pf::AbstractVector{<:Real}
    Xb::AbstractVector{<:Real}
    Pb::AbstractVector{<:Real}
end

struct SpinPLDMSys <: Systems.SpinMappedSystem
    transform::Type{<:Systems.SWTransform}
    h::AbstractMatrix{<:Complex}
    ρ₀::AbstractMatrix{<:Complex}
    R²::Float64
    γₛ::Float64
    d::Integer
    bath::Solvents.Solvent
    nsamples::Integer
end
function SpinPLDMSys(; transform::Type{<:Systems.SWTransform},
                     Hamiltonian::AbstractMatrix{<:Complex},
                     ρ₀::AbstractMatrix{<:Complex},
                     bath::Solvents.Solvent, nsamples::Integer)
    @assert nsamples == length(bath)
    d = size(Hamiltonian, 1)
    SpinPLDMSys(transform, Hamiltonian, ρ₀,
                Systems.R²(transform, d), Systems.γ(transform, d),
                d, bath, nsamples)
end

function Base.iterate(sys::SpinPLDMSys, state=1)
    state > sys.nsamples && return nothing

    bathps, _ = iterate(sys.bath, state)

    Xf, Pf = Systems.sample_XP(sys)
    Xb, Pb = Systems.sample_XP(sys)

    (SpinPLDMSysPhaseSpace(Xf, Pf, Xb, Pb), bathps), state+1
end
Base.eltype(::SpinPLDMSys) = SpinPLDMSysPhaseSpace
Base.length(s::SpinPLDMSys) = s.nsamples
Base.firstindex(::SpinPLDMSys) = 1
Base.getindex(s::SpinPLDMSys, n::Integer) = iterate(s, n)[1]



function Systems.transform_op(sys::SpinPLDMSys, op::Union{AbstractVector,AbstractMatrix},
                              ps::SpinPLDMSysPhaseSpace, path::Symbol)
    @assert path ∈ [ :forward, :backward ]
    if path == :forward
        Systems.transform_op(sys, op, ps.Xf, ps.Pf)
    else
        Systems.transform_op(sys, op, ps.Xb, ps.Pb)
    end
end

function transform_kernel(sys::SpinPLDMSys,
                          X₀::Vector{<:Real}, P₀::Vector{<:Real},
                          Xₜ::Vector{<:Real}, Pₜ::Vector{<:Real},
                          U::AbstractMatrix{<:Complex})
    dual = Systems.dual(sys.transform)
    rescale = Systems.rescale_factor(sys.transform, dual, sys.d)
    γs̄ = Systems.γ(dual, sys.d)

    X̄₀ = X₀ * rescale
    P̄₀ = P₀ * rescale
    X̄ₜ = Xₜ * rescale
    P̄ₜ = Pₜ * rescale

    0.5 * ((X̄ₜ + im * P̄ₜ) * (X̄₀ + im * P̄₀)' - γs̄ * U)
end

function build_ρ!(sys::SpinPLDMSys, sps0::SpinPLDMSysPhaseSpace,
                  Xf::Vector{Float64}, Pf::Vector{Float64},
                  Xb::Vector{Float64}, Pb::Vector{Float64},
                  ρ::AbstractMatrix{<:Complex}, U::AbstractMatrix{<:Complex})
    wf = transform_kernel(sys, sps0.Xf, sps0.Pf, Xf, Pf, U)
    wb = transform_kernel(sys, sps0.Xb, sps0.Pb, Xb, Pb, U)
    ρ[:,:] = sys.d^2 * (wf * sys.ρ₀ * wb' + wb * sys.ρ₀ * wf') / 2
end

function propagate_trajectory(sys::SpinPLDMSys,
                              sps0::SpinPLDMSysPhaseSpace,
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
    U = diagm(ones(ComplexF64, d))

    @views build_ρ!(sys, sps0, Xf, Pf, Xb, Pb, ρ[1,:,:], U)

    dt2 = dt / 2
    bs = sys.bath
    A = zeros(d,d)
    LU = zeros(ComplexF64, 2d,2d)
    s̄ₛc = similar.(bs.c)
    Systems.Fbath!(sys, sps0, s̄ₛc)
    sps = SpinPLDMSysPhaseSpace(Xf, Pf, Xb, Pb)
    @inbounds for t in 2:ntimes+1
        Solvents.propagate_forced_bath!(bs, bps₀, bpsₙ, s̄ₛc, dt2, 1)

        sinA, cosA = Systems.get_propagator(sys, bpsₙ, A, dt)
        Systems.apply_propagator!(sys, sps, sinA, cosA, buf)
        # A now has H(q, p) * dt.  U requires exp(-im * A).
        U = (cosA - im .* sinA) * U

        Systems.Fbath!(sys, sps, s̄ₛc)
        Solvents.propagate_forced_bath!(bs, bpsₙ, bps₀, s̄ₛc, dt2, 1)

        @views build_ρ!(sys, sps0, Xf, Pf, Xb, Pb, ρ[t,:,:], U)
    end

    ρ
end

function propagate_trajectories(sys::SpinPLDMSys, dt::Real, ntimes::Integer;
                                output::Union{Nothing,HDF5.Group}=nothing,
                                verbose::Bool=false, kwargs...)
    ρ = isnothing(sys.ρ₀) ? nothing : zeros(ComplexF64, ntimes+1,sys.d,sys.d)
    outputρ = if !isnothing(ρ) && haskey(kwargs, :outgroup)
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
            isnothing(ρ) || (ρ .+= ρᵢ)
            verbose && ndone % nthreads == 0 &&
                @info "Trajectories complete: $(100ndone / length(sys))%"
        end
    end
    @info "All trajectories complete"
    @info "Time taken = $(round(stats.time; digits=3)) sec; memory allocated = $(round(stats.bytes / 1e9; digits=3)) GB; gc time = $(round(stats.gctime; digits=3)) sec"

    if !isnothing(ρ)
        ρ ./= length(sys)
        if !isnothing(outputρ)
            outputρ["rho"] = ρ
            flush(outputρ)
        end
    end

    ρ
end

"""
    propagate(; Hamiltonian::Matrix{<:Complex}, Jw::Vector{T},
              β::Real, num_osc::Vector{<:Integer}, svec::Matrix{<:Real},
              ρ0::Matrix{<:Complex}, dt::Real,
              ntimes::Real, transform::Type{<:Systems.SWTransform},
              nmc::Integer, verbose::Bool=false,
              kwargs...) where {T<:SpectralDensities.SpectralDensity}

Propagate the system using the spin-mapped PLDM method.

Arguments:
- `ρ0`: initial reduced density matrix
- `Hamiltonian`: the Hamiltonian of the sub-system
- `Jw`: list of spectral densities
- `β`: the inverse temperature of the bath
- `num_osc`: the number of oscillator for each bath
- `svec`: diagonal elements of system operators through which the
  corresponding baths interact
- `transform`: the Stratonovich–Weyl transformation to use for the
  Hamiltonian
- `dt`: the time step for the propagation
- `nmc`: the number of Monte-Carlo samples

Propagate the density matrix using a partially linearised propagator
for the system-bath problem, using the given Stratonovich–Weyl
transform for the Hamiltonian.
"""
function propagate(; Hamiltonian::Matrix{<:Complex}, Jw::Vector{T},
                   β::Real, num_osc::Vector{<:Integer}, svec::Matrix{<:Real},
                   ρ0::Matrix{<:Complex}, dt::Real,
                   ntimes::Real, transform::Type{<:Systems.SWTransform},
                   nmc::Integer, verbose::Bool=false,
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
    sys = SpinPLDMSys(; transform, Hamiltonian, ρ₀=ρ0, bath, nsamples=nmc)

    propagate_trajectories(sys, dt, ntimes; verbose, kwargs...)
end

end
