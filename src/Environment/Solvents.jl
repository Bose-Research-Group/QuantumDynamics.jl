"General description of different types of solvent."
module Solvents

using Distributions: MvNormal
using LinearAlgebra: Diagonal, diagm

"""Abstract type for every solvent.
Every solvent needs to implement `Base.iterate`, which returns the
next sample of the phase-space, and a `propagate_trajectory` method
which propagates a given phasespace point using classical mechanics.
"""
abstract type Solvent end

"""Abstract type for all phase spaces.
Each `Solvent` has an associated phase-space.
"""
abstract type PhaseSpace end

struct HarmonicPhaseSpace <: PhaseSpace
    q::Vector{Vector{Float64}}
    p::Vector{Vector{Float64}}
end

struct HarmonicBath <: Solvent
    β::Real
    ω::Vector{Vector{Float64}}
    c::Vector{Vector{Float64}}
    s::Vector{Vector{Float64}}
    distq::Vector{MvNormal}
    distp::Vector{MvNormal}
    nsamples::Integer
    nbaths::Integer
end
function HarmonicBath(; β::Real, ω::Vector{Vector{Float64}},
                         c::Vector{Vector{Float64}},
                         svecs::Vector{Vector{Float64}},
                         nsamples::Integer)
    @assert length(c) == length(ω)
    @assert length(c) == length(svecs)

    distq = Vector{MvNormal}(undef, length(ω))
    distp = Vector{MvNormal}(undef, length(ω))

    for b in 1:length(ω)
        cth = coth.(0.5 * ω[b] * β)
        distq[b] = MvNormal(Diagonal(cth ./ ω[b] / 2))
        distp[b] = MvNormal(Diagonal(cth .* ω[b] / 2))
    end

    HarmonicBath(β, ω, c, svecs, distq, distp, nsamples, length(ω))
end

function Base.iterate(bath::HarmonicBath, state=1)
    state > bath.nsamples && return nothing

    (HarmonicPhaseSpace(
        map(dist -> rand(dist), bath.distq),
        map(dist -> rand(dist), bath.distp)),
     state+1)
end
Base.eltype(::HarmonicBath) = HarmonicPhaseSpace
Base.length(b::HarmonicBath) = b.nsamples
Base.firstindex(::HarmonicBath) = 1
Base.getindex(b::HarmonicBath, n::Integer) = iterate(b, n)[1]

@doc raw"""
    propagate_forced_bath(bath::HarmonicBath, bps::HarmonicPhaseSpace,
                          f::Vector{Vector{Float64}}, dt::Real, ntimes::Integer)

Propagate the `bath` subject to constant force `f` from the system.

This assumes the Hamiltonian of the oscillator is of the form
``H = \frac{p^2}{2} + \frac{\omega^2 q^2}{2} - f q.``
"""
function propagate_forced_bath(bath::HarmonicBath, bps::HarmonicPhaseSpace,
                               f::Vector{Vector{Float64}}, dt::Real, ntimes::Integer)
    q = similar.(bps.q)
    p = similar.(bps.p)
    Σcω² = [ 0.5mapreduce((c, ω) -> c^2 / ω^2, +, bath.c[b], bath.ω[b]) for b in eachindex(bath.c) ]
    E_bath = [ mapreduce(b -> Σcω²[b] * bath.s[b][j]^2, +, eachindex(bath.s)) for j in 1:length(bath.s[1]) ]
    energies = zeros(ntimes+1, length(E_bath))

    for t in 1:ntimes+1
        energies[t,:] .= E_bath
    end
    for b in eachindex(bath.s)
        energies[1,:] .-= bath.s[b] * sum(bath.c[b] .* bps.q[b])
    end

    @inbounds for t in 1:ntimes
        for b in eachindex(bps.q)
            sinωt = @. sin(bath.ω[b] * t * dt)
            cosωt = @. cos(bath.ω[b] * t * dt)
            disp = @. f[b] / bath.ω[b]^2
            @. q[b] =  (bps.q[b] - disp) * cosωt + bps.p[b] * sinωt / bath.ω[b] + disp
            @. p[b] = -(bps.q[b] - disp) * bath.ω[b] * sinωt + bps.p[b] * cosωt
           energies[t+1,:] .-= bath.s[b] * sum(bath.c[b] .* q[b])
        end
    end

    energies, HarmonicPhaseSpace(q, p)
end

function propagate_forced_bath!(bath::HarmonicBath, bps₀::HarmonicPhaseSpace, bpsₙ::HarmonicPhaseSpace, f::Vector{Vector{Float64}}, dt::Real, ntimes::Integer)
    dt *= ntimes

    @inbounds for b in eachindex(bps₀.q)
        for i in eachindex(bps₀.q[b])
            sinωt, cosωt = sincos(bath.ω[b][i] * dt)
            disp = f[b][i] / bath.ω[b][i]^2
            bpsₙ.q[b][i] =  (bps₀.q[b][i] - disp) * cosωt + bps₀.p[b][i] * sinωt / bath.ω[b][i] + disp
            bpsₙ.p[b][i] = -(bps₀.q[b][i] - disp) * bath.ω[b][i] * sinωt + bps₀.p[b][i] * cosωt
        end
    end
end

"""
    bath_force(bath::Solvent, state::PhaseSpace, n::Integer)

Return the `n`th bath's (self-)force at the phase space point given by `state`.
"""
function bath_force(bath::Solvent, state::PhaseSpace, n::Integer) end

bath_force(bath::HarmonicBath, state::HarmonicPhaseSpace, n::Integer) = -bath.ω[n]^2 * state.q[n]

@doc raw"""
    propagate_forced_bath(bath::Solvent, state::PhaseSpace,
                          f::Vector{Vector{Float64}}, dt::Real,
                          ntimes::Integer)

Propagate `bath` subject to constant force `f` from the system.

The Hamiltonian for a single bath is assumed to be of the form
``H = \sum_i \frac{p_i^2}{2} + V(\{q_i\}) - f_i q_i``
where `V(\{q_i\})` is the state-independent potential of the bath.

The time step of the propagation is given by `dt`, and the number of
Verlet steps by `ntimes`.
"""
function propagate_forced_bath(bath::Solvent, state::PhaseSpace,
                               f::Vector{Vector{Float64}}, dt::Real,
                               ntimes::Integer)
    q = similar.(state.q)
    p = similar.(state.p)

    @inbounds for _ in 1:ntimes
        for b in eachindex(state.q)
            @. p[b] += 0.5 * (bath_force(bath, state, b) + f[b]) * dt
            @. q[b] += p[b] * dt
        end
        state = eltype(bath)(q, p)
        for b in eachindex(state.p)
            @. p[b] += 0.5 * (bath_force(bath, state, b) + f[b]) * dt
        end
    end

    eltype(bath)(q, p)
end

function get_Vint(ps::HarmonicPhaseSpace, hb::HarmonicBath, t::Float64)
    Ham = zero(diagm(hb.s[1]))
    for (q, p, ω, c, s) in zip(ps.q, ps.p, hb.ω, hb.c, hb.s)
        Ham += - sum((q .* cos.(ω * t) + p .* sin.(ω * t) ./ ω) .* c) * diagm(s)
    end
    Ham
end

end
