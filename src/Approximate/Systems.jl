"General description of the system, particularly for trajectory based methods."
module Systems

using ..Solvents: PhaseSpace
using LinearAlgebra: diag, ⋅, Diagonal

"""Abstract phase space for methods that use a linearised semiclassical approximation."""
abstract type LinearisedSysPhaseSpace <: PhaseSpace end

"""Abstract phase space for methods that use a partially linearised semiclassical approximation."""
abstract type PartialLinearisedSysPhaseSpace <: PhaseSpace end

"""Abstract type for methods that propagate system and bath together.
All such systems must implement `Base.iterate` which returns the next
sample point of the system and the solvent."""
abstract type CompositeSystem end

"""Abstract type for mapping Hamiltonian methods."""
abstract type MappedSystem <: CompositeSystem end

"""
    γ(::MappedSystem)

Return the zero-point energy parameter for system with mapped Hamiltonian.
"""
function γ(::MappedSystem) end

@doc raw"""
    transform_op(sys::MappedSystem, op::AbstractVector{<:Number},
                 X::AbstractVector{<:Real}, P::AbstractVector{<:Real})

Return the mapped form of the diagonal operator `op`.

This function assumes that the mapping formalism can be put in the
following general form for an arbitrary operator `Ω`
```math
\Omega_{\rm mapped} = \sum_j \Omega_{jj} \frac{X_j^2 + P_j^2 - \gamma}{2}
```
where `γ` is the zero-point energy parameter, which is obtained using
the system `sys`'s [`Systems.γ`](@ref) method.
"""
function transform_op(sys::MappedSystem, op::AbstractVector{<:Number},
                      X::AbstractVector{<:Real}, P::AbstractVector{<:Real})
    γₛ = γ(sys)
    @inbounds 0.5mapreduce((op,X,P) -> op * (X^2 + P^2 - γₛ), +, op, X, P)
end

@doc raw"""
    transform_op(sys::MappedSystem, op::AbstractMatrix{<:Number},
                 X::AbstractVector{<:Real}, P::AbstractVector{<:Real})

Return the mapped form of the general operator `op`.

This function assumes that the mapped operator has the following
general form
```math
\Omega_{\rm mapped} = \sum_jk \Omega_{jk} \frac{(X_j - i P_j)(X_k + i P_k) - \delta_{jk} \gamma}{2}
```
where `γ` and `δⱼₖ` are the zero-point energy parameter and the
Kronecker delta respectively.  The former is obtained using the system
`sys`'s [`Systems.γ`](@ref) method.
"""
function transform_op(sys::MappedSystem, op::AbstractMatrix{<:Number},
                      X::AbstractVector{<:Real}, P::AbstractVector{<:Real})
    opt = transform_op(sys, diag(op), X, P)

    @inbounds for n = 1:sys.d, m = n+1:sys.d
        opt += 0.5 * op[n,m] * (X[n] - im * P[n]) * (X[m] + im * P[m])
        opt += 0.5 * op[m,n] * (X[m] - im * P[m]) * (X[n] + im * P[n])
    end

    opt
end

"""
    transform_op(sys::MappedSystem, op::Union{AbstractMatrix,AbstractVector},
                 ps::PartialLinearisedSysPhaseSpace, path::Symbol)

Return the mapped form of operator `op` for the phase space point in specified path.

The argument `path` can be either `:forward` or `:backward` depending
on whether the operator should be mapped for the forward or backward
path respectively.
"""
function transform_op(sys::MappedSystem, op::Union{AbstractMatrix,AbstractVector}, ps::PartialLinearisedSysPhaseSpace, path::Symbol) end

@doc raw"""
        Fbath!(sys::MappedSystem, ps::LinearisedSysPhaseSpace, f::Vector{<:AbstractVector{<:Real}})

Calculate the system `sys`'s force on the bath and write it to `f`.

This function assumes that the system-bath interaction is of the form
```math
H_{sb} = \sum\limits_b^{N_{\rm bath}} \sum\limits_j^{N_{\rm osc, j}} - c_{bj} \omega_{bj} s_b.
```
"""
function Fbath!(sys::MappedSystem, ps::LinearisedSysPhaseSpace, f::Vector{<:AbstractVector{<:Real}})
    @inbounds for b in eachindex(sys.bath.c)
        sₘ = transform_op(sys, sys.bath.s[b], ps)
        @. f[b] = sₘ * sys.bath.c[b]
    end
end

@doc raw"""
        Fbath!(sys::MappedSystem, ps::PartialLinearisedSysPhaseSpace, f::Vector{<:AbstractVector{<:Real}})

Write the average force exerted by the system `sys` on the bath to `f`.

This method calculates the force as the average of the force from the
forward and the backward path of the system.
"""
function Fbath!(sys::MappedSystem, ps::PartialLinearisedSysPhaseSpace, f::Vector{<:AbstractVector{<:Real}})
    @inbounds for b in eachindex(sys.bath.c)
        s̄ₘ = 0.5(transform_op(sys, sys.bath.s[b], ps, :forward) +
                 transform_op(sys, sys.bath.s[b], ps, :backward))
        @. f[b] = s̄ₘ * sys.bath.c[b]
    end
end

### Spin-mapping methods.

"Abstract type for all Stratonovich–Weyl transforms."
abstract type SWTransform end

abstract type WTransform <: SWTransform end
abstract type QTransform <: SWTransform end
abstract type PTransform <: SWTransform end

R²(::Type{WTransform}, d::Integer) = 2sqrt(d+1)
R²(::Type{QTransform}, d::Integer) = 2.0
R²(::Type{PTransform}, d::Integer) = 2.0 * (d+1)

γ(t::Type{<:SWTransform}, d::Integer) = (R²(t, d) - 2.0) / d
γ(::Type{QTransform}, d::Integer) = 0.0

dual(W::Type{WTransform}) = W
dual(::Type{PTransform}) = QTransform
dual(::Type{QTransform}) = PTransform

rescale_factor(s::Type{<:SWTransform}, s̄::Type{<:SWTransform}, d::Integer) = sqrt(R²(s̄, d) / R²(s, d))
rescale_factor(::Type{WTransform}, ::Type{WTransform}, ::Integer) = 1

"""Abstract type for all spin-mapped systems.
It should contain the following elements always:
  - `transform`: the Stratonovich–Weyl transform
  - `d`: the dimensionality of the system
  - `γₛ`: the zero-point energy parameter of the Stratonovich–Weyl transform
  - `R²`: the squared-radius parameter for the Stratonovich–Weyl transform
"""
abstract type SpinMappedSystem <: MappedSystem end

γ(s::SpinMappedSystem) = s.γₛ

"""
     sample_XP(sys::SpinMappedSystem)

Draw a random sample of the pseudo position and momentum for `sys`.
This draws a sample from the hypersphere surface associated with the
system's Stratonovich–Weyl transform.
"""
function sample_XP(sys::SpinMappedSystem)
    R = sqrt(sys.R²)
    X = randn(sys.d)
    P = randn(sys.d)

    sqΣ = sqrt(sum(X.^2 .+ P.^2))

    X * R / sqΣ, P * R / sqΣ
end

### Common system dof propagation routines.

"""
    get_propagator(sys::MappedSystem, bps::Solvents.PhaseSpace,
                   A::AbstractMatrix, dt::Real)

Get the "propagator" for the system with bath coordinates given by `bps`.

This routine does not actually return the full propagator but returns
two matrices which can be put together to get the full Liouvillian.
Suppose `D` and `E` are the two matrices that are returned in order,
then
```math
\mathcal{L} = \begin{pmatrix}
D & -E
E & D
\end{pmatrix}
```

`A` is the work matrix whose data will be overwritten, `dt` is the
time step.
"""
function get_propagator(sys::MappedSystem, bps::PhaseSpace,
                        A::AbstractMatrix, dt::Real)
    A .= sys.h
    for b in eachindex(sys.bath.c)
        A .-= @views (sys.bath.c[b] ⋅ bps.q[b]) * Diagonal(sys.bath.s[b])
    end
    A .*= dt
    sincos(A)
end

function apply_propagator!(sys::MappedSystem, sps::LinearisedSysPhaseSpace, sinA::AbstractMatrix, cosA::AbstractMatrix, buf::AbstractVector{<:Real})
    buf .= sps.X
    sps.X .= cosA * sps.X - sinA * sps.P
    sps.P .= sinA * buf + cosA * sps.P
end

function apply_propagator!(sys::MappedSystem, sps::PartialLinearisedSysPhaseSpace, sinA::AbstractMatrix, cosA::AbstractMatrix, buf::AbstractVector{<:Real})
    buf .= sps.Xf
    sps.Xf .= cosA * sps.Xf - sinA * sps.Pf
    sps.Pf .= sinA * buf + cosA * sps.Pf
    buf .= sps.Xb
    sps.Xb .= cosA * sps.Xb - sinA * sps.Pb
    sps.Pb .= sinA * buf + cosA * sps.Pb
end

end
