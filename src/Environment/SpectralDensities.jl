"Collection of spectral densities commonly used to describe solvents."
module SpectralDensities

using DelimitedFiles
using LinearAlgebra
using ..Utilities

const references = """
- Makri, N. The Linear Response Approximation and Its Lowest Order Corrections: An Influence Functional Approach. The Journal of Physical Chemistry B 1999, 103 (15), 2823–2829. https://doi.org/10.1021/jp9847540.
- Bose, A. Zero-Cost Corrections to Influence Functional Coefficients from Bath Response Functions. The Journal of Chemical Physics 2022, 157 (5), 054107. https://doi.org/10.1063/5.0101396."""

"""
    SpectralDensity
Abstract base type for all spectral densities.
"""
abstract type SpectralDensity end
"""
    ContinuousSpectralDensity <: SpectralDensity
Abstract base type for all continuous spectral densities.
"""
abstract type ContinuousSpectralDensity <: SpectralDensity end
"""
    DiscreteOscillators <: SpectralDensity
Describes a bath of discrete oscillators. Contains:
- `ω`: frequencies of the different oscillators
- `jw`: spectral density for each of the oscillators
"""
struct DiscreteOscillators <: SpectralDensity
    ω::Vector{Float64}
    jw::Vector{Float64}
    classical::Bool
end
function read_discrete_jw(filename; skipstart=0, classical=false)
    w_jw = readdlm(filename; skipstart)
    DiscreteOscillators(w_jw[:, 1], w_jw[:, 2], classical)
end
function read_discrete_jw_over_w(filename; skipstart=0, classical=false)
    w_jw = readdlm(filename; skipstart)
    DiscreteOscillators(w_jw[:, 1], w_jw[:, 2] .* w_jw[:, 1], classical)
end
function read_huang_rhys(filename; skipstart=0, classical=false)
    w_S = readdlm(filename; skipstart)
    w_S[:, 2] .*= π .* (w_S[:, 1]) .^ 2
    DiscreteOscillators(w_S[:, 1], w_S[:, 2], classical)
end

"""
    tabulate(sd::DiscreteOscillators, full_real::Bool=true)
Returns `sd.ω` and `sd.jw`.
"""
function tabulate(sd::DiscreteOscillators, full_real::Bool=true)
    if full_real
        [-reverse(sd.ω); sd.ω], [-reverse(sd.jw); sd.jw]
    else
        sd.ω, sd.jw
    end
end

"""
    AnalyticalSpectralDensity <: ContinuousSpectralDensity
Abstract base type for all model analytical spectral densities. An analytical spectral density, `J`, can be evaluated at a frequency, `ω`, as `J(ω)`.
"""
abstract type AnalyticalSpectralDensity <: ContinuousSpectralDensity end
(sd::AnalyticalSpectralDensity)(ω::Real) = evaluate(sd, ω)
eval_spectrum(sd::AnalyticalSpectralDensity, ω::Real, β::Real) = ω == 0.0 ? eval_spectrum_at_zero(sd) : 2.0 * sd(ω) / (1 - exp(-β * ω))

"""
    ExponentialCutoff <: AnalyticalSpectralDensity
Model spectral density with an exponential cutoff of the form:

``J(ω) = \\frac{2π}{Δs^2} ξ \\frac{ω^n}{ω_c^{n-1}} \\exp\\left(-\\frac{|ω|}{ωc}\\right)``

where `Δs` is the distance between the two system states, `ξ` is the dimensionless Kondo parameter, and `ωc` is the cutoff frequency. The model is Ohmic if `n = 1`, sub-Ohmic if `n < 1`, and super-Ohmic if `n > 1`.

The struct contains:
- `ξ`: Kondo parameter
- `ωc`: cutoff frequency
- `Δs`: the distance between the two states
- `n`: power of the polynomial
- `ωmax`: when discretized the points would lie in the symmetric interval, [-ωmax, ωmax]
- `npoints`: number of points of discretization
- `classical`: is the spectral density describing a classical bath?
"""
struct ExponentialCutoff <: AnalyticalSpectralDensity
    ξ::Float64
    ωc::Float64
    Δs::Float64
    n::Float64
    ωmax::Float64
    npoints::Int64
    classical::Bool
end
ExponentialCutoff(; ξ::Float64, ωc::Float64, n=1.0, Δs=2.0, ωmax=30 * ωc, classical=false, npoints=10000) = ExponentialCutoff(ξ, ωc, Δs, n, ωmax, npoints, classical)
evaluate(sd::ExponentialCutoff, ω::T) where {T<:AbstractFloat} = T(2π) / sd.Δs^2 * sd.ξ * sign(ω) * abs(ω)^sd.n * sd.ωc^(1 - sd.n) * exp(-abs(ω) / sd.ωc)
eval_spectrum_at_zero(sd::ExponentialCutoff) = sd.n == 1 ? 2.0 * 2π / sd.Δs^2 * sd.ξ : 0

"""
    DrudeLorentz <: AnalyticalSpectralDensity
Model Drude-Lorentz spectral density of the form:

``J(ω) = \\frac{2λ}{Δs^2} \\frac{ω γ}{ω^2 + γ^2}``

where `Δs` is the distance between the two system states.

The struct contains:
- `γ`: cutoff frequency
- `λ`: reorganization energy
- `Δs`: the distance between the two states
- `ωmax`: when discretized the points would lie in the symmetric interval, [-ωmax, ωmax]
- `npoints`: number of points of discretization
- `classical`: is the spectral density describing a classical bath?
"""
struct DrudeLorentz <: AnalyticalSpectralDensity
    λ::Float64
    γ::Float64
    Δs::Float64
    ωmax::Float64
    npoints::Int64
    classical::Bool
end
DrudeLorentz(; λ::T, γ::T, Δs=2.0, ωmax=1000 * γ, classical=false, npoints=10000) where {T<:AbstractFloat} = DrudeLorentz(λ, γ, Δs, ωmax, npoints, classical)
evaluate(sd::DrudeLorentz, ω::Real) = 2 * sd.λ / sd.Δs^2 * sign(ω) * abs(ω) * sd.γ / (abs(ω)^2 + sd.γ^2)
eval_spectrum_at_zero(sd::DrudeLorentz) = 2 * 2 * sd.λ / sd.Δs^2 * sd.γ
Δk_target(sd::SpectralDensities.DrudeLorentz, β) = 2 * sd.λ / (sd.Δs^2 * sd.γ * β)

"""
    UnderdampedBrownian <: AnalyticalSpectralDensity
Model Underdamped Brownian spectral density of the form:

``J(ω) = \\frac{2λ}{Δs^2} \\frac{ω γ ω0^2}{(ω^2 - ω0^2)^2 + γ^2 ω^2}``

where `Δs` is the distance between the two system states.

The struct contains:
- `λ`: reorganization energy
- `ω0`: natural frequency
- `γ`: damping rate
- `Δs`: the distance between the two states
- `ωmax`: when discretized the points would lie in the symmetric interval, [-ωmax, ωmax]
- `npoints`: number of points of discretization
- `classical`: is the spectral density describing a classical bath?
"""
struct UnderdampedBrownian <: AnalyticalSpectralDensity
    λ::Float64
    ω0::Float64
    γ::Float64
    Δs::Float64
    ωmax::Float64
    npoints::Int64
    classical::Bool
end
UnderdampedBrownian(; λ::T, ω0::T, γ::T, Δs=2.0, ωmax=1000 * max(γ, ω0), classical=false, npoints=10000) where {T<:AbstractFloat} = UnderdampedBrownian(λ, ω0, γ, Δs, ωmax, npoints, classical)
evaluate(sd::UnderdampedBrownian, ω::Real) = 2 * sd.λ / sd.Δs^2 * sign(ω) * abs(ω) * sd.γ * sd.ω0^2 / ((ω^2 - sd.ω0^2)^2 + sd.γ^2 * ω^2)
eval_spectrum_at_zero(sd::UnderdampedBrownian) = 4 * sd.λ * sd.γ / (sd.Δs^2 * sd.ω0^2)
Δk_target(sd::SpectralDensities.UnderdampedBrownian, β) = 2 * sd.λ * sd.γ / (sd.Δs^2 * sd.ω0^2 * β)

"""
    ExponentialDecomposition
Contains the data for the representation of a bath correlation function as a sum of exponentials with complex rates. Used for HEOM.

The struct contains:
- `ν`: rates
- `c`: coefficients
- `ctilde`: coefficients of the complex conjugate rate mode
- `scale`: HEOM scaling factor
"""
struct ExponentialDecomposition
    ν::Vector{ComplexF64}
    c::Vector{ComplexF64}
    ctilde::Vector{ComplexF64}
    scale::Vector{Float64}

    function ExponentialDecomposition(ν,c,ctilde)
        @assert length(ν) == length(c) == length(ctilde)
        scale = sqrt.(abs.(c .* ctilde))
        new(ComplexF64.(ν), ComplexF64.(c), ComplexF64.(ctilde), scale)
    end
end
imaginary_response_decomposition(sd::SpectralDensity, num_modes::Int) = error("Imaginary response decomposition not implemented for $(typeof(sd)).")
matsubara_decomposition(sd::SpectralDensity, num_modes::Int, β::AbstractFloat) = error("Matsubara decomposition not implemented for $(typeof(sd)).")
pade_decomposition(sd::SpectralDensity, num_modes::Int, β::AbstractFloat) = error("Pade decomposition not implemented for $(typeof(sd)).")

"""
    matsubara_decomposition(sd::DrudeLorentz, num_modes::Int, β::AbstractFloat)

Implements the Matsubara decomposition for the Drude-Lorentz spectral density.
Returns the decay rates, `γ`, and the expansion coefficients, `c`.
"""
function matsubara_decomposition(sd::DrudeLorentz, num_modes::Int, β::AbstractFloat)
    elem_type = typeof(sd.λ)
    γ = zeros(Complex{elem_type}, num_modes + 1)
    c = zeros(Complex{elem_type}, num_modes + 1)
    γ[1] = sd.γ
    c[1] = sd.λ * sd.γ / sd.Δs^2 * (cot(β * sd.γ / (2 * one(elem_type))) - 1im)
    for k = 2:num_modes+1
        γ[k] = 2 * (k - 1) * elem_type(π) / β
        c[k] = 4 * sd.λ / sd.Δs^2 * sd.γ / β * γ[k] / (γ[k]^2 - sd.γ^2)
    end

    ExponentialDecomposition(γ, c, conj.(c))
end
imaginary_response_decomposition(sd::DrudeLorentz, num_modes::Int) = ExponentialDecomposition([sd.γ + 0.0im], [-1im * sd.λ * sd.γ / sd.Δs^2], [1im * sd.λ * sd.γ / sd.Δs^2])
function matsubara_decomposition(sd::UnderdampedBrownian, num_modes::Int, β::AbstractFloat)
    elem_type = typeof(sd.λ)
    Ω = sqrt(complex(sd.ω0^2 - sd.γ^2 / 4))
    A = sd.λ * sd.ω0^2 / (2 * sd.Δs^2 * Ω)

    ν = zeros(Complex{elem_type}, num_modes + 2)
    c = zeros(Complex{elem_type}, num_modes + 2)

    ν[1] = sd.γ / 2 - 1im * Ω
    ν[2] = sd.γ / 2 + 1im * Ω
    c[1] = -A * (1 + coth(β * (-Ω - 1im * sd.γ / 2) / 2))
    c[2] =  A * (1 + coth(β * ( Ω - 1im * sd.γ / 2) / 2))

    for k = 1:num_modes
        νk = 2 * k * elem_type(π) / β
        ν[k+2] = νk
        c[k+2] = -4 * sd.λ * sd.γ * sd.ω0^2 * νk / (β * sd.Δs^2 * ((νk^2 + sd.ω0^2)^2 - sd.γ^2 * νk^2))
    end

    ctilde = similar(c)
    ctilde[1] = conj(c[2])   # crossed pairing — ν[1],ν[2] are a genuine conjugate
    ctilde[2] = conj(c[1])   # pair, same convention as imaginary_response_decomposition
    for k = 1:num_modes
        ctilde[k+2] = conj(c[k+2])  # real ν here, standard self-pairing
    end

    ExponentialDecomposition(ν, c, ctilde)
end
function imaginary_response_decomposition(sd::UnderdampedBrownian, num_modes::Int)
    Ω = sqrt(complex(sd.ω0^2 - sd.γ^2/4))
    ν = ComplexF64[ sd.γ/2 - 1im*Ω, sd.γ/2 + 1im*Ω ]
    A = sd.λ * sd.ω0^2 / (2 * sd.Δs^2 * Ω)
    c = ComplexF64[ -A, A ]
    # crossed conjugate coefficients
    ctilde = ComplexF64[ conj(c[2]), conj(c[1]) ]
    ExponentialDecomposition(ν, c, ctilde)
end

"""
    pade_decomposition(sd::DrudeLorentz, num_modes::Int, β::AbstractFloat)

Implements the [N-1/N] Padé spectrum decomposition for the Drude-Lorentz spectral density.
Returns the decay rates, `γ`, and the expansion coefficients, `c`.
"""
function pade_decomposition(sd::DrudeLorentz, num_modes::Int, β::AbstractFloat)
    elem_type = typeof(sd.λ)
    γ = zeros(Complex{elem_type}, num_modes + 1)
    c = zeros(Complex{elem_type}, num_modes + 1)
    
    # Padé [N-1/N] poles (η) and residues (κ) for the Bose-Einstein distribution
    η, κ = get_pade_poles_residues(num_modes, elem_type)
    
    γ[1] = sd.γ
    c[1] = sd.λ * sd.γ / sd.Δs^2 * (cot(β * sd.γ / (2 * one(elem_type))) - 1im)

    for k = 1:num_modes
        γ[k+1] = η[k] / β
        c[k+1] = (4 * sd.λ * sd.γ) / (β * sd.Δs^2) * (κ[k] * γ[k+1] / (γ[k+1]^2 - sd.γ^2))
    end

    ExponentialDecomposition(γ, c, conj.(c))
end
function pade_decomposition(sd::UnderdampedBrownian, num_modes::Int, β::AbstractFloat)
    elem_type = typeof(sd.λ)
    Ω = sqrt(complex(sd.ω0^2 - sd.γ^2 / 4))
    A = sd.λ * sd.ω0^2 / (2 * sd.Δs^2 * Ω)

    ν = zeros(Complex{elem_type}, num_modes + 2)
    c = zeros(Complex{elem_type}, num_modes + 2)

    ν[1] = sd.γ / 2 - 1im * Ω
    ν[2] = sd.γ / 2 + 1im * Ω
    c[1] = -A * (1 + coth(β * (-Ω - 1im * sd.γ / 2) / 2))
    c[2] =  A * (1 + coth(β * ( Ω - 1im * sd.γ / 2) / 2))

    η, κ = get_pade_poles_residues(num_modes, elem_type)  # reused as-is — universal to coth(βω/2)
    for k = 1:num_modes
        νk = η[k] / β
        ν[k+2] = νk
        c[k+2] = κ[k] * (-4 * sd.λ * sd.γ * sd.ω0^2 * νk / (β * sd.Δs^2 * ((νk^2 + sd.ω0^2)^2 - sd.γ^2 * νk^2)))
    end

    ctilde = similar(c)
    ctilde[1] = conj(c[2])
    ctilde[2] = conj(c[1])
    for k = 1:num_modes
        ctilde[k+2] = conj(c[k+2])
    end

    ExponentialDecomposition(ν, c, ctilde)
end

"""
    get_pade_poles_residues(N::Int, T::Type)

Constructs the specific tridiagonal matrix whose eigenvalues and 
eigenvectors define the [N-1/N] Padé poles and residues.
"""
function get_pade_poles_residues(N::Int, T::Type)
    N == 0 && return T[], T[]

    b(m::Int64; symmtype="boson") = (symmtype == "boson") ? (2m+1) : (2m-1)

    d = [1 / sqrt(b(j) * b(j+1)) for j=1:2N-1]
    C = SymTridiagonal(zeros(2N), d)
    vals, vecs = eigen(C)

    idx = findall(vals .> 100 * eps(Float64))
    ξ = 2 ./ vals[idx]
    sort!(ξ)

    Ctilde = SymTridiagonal(zeros(2N-1), d[2:end])
    vals, vecs = eigen(Ctilde)
    idx = findall(vals .> 100 * eps(Float64))
    ζ = 2 ./ vals[idx]

    η = ones(N) * N * b(N+1) / 2
    for j = 1:N
        for k = 1:N-1
            η[j] *= ζ[k]^2 - ξ[j]^2
            if k != j
                η[j] /= ξ[k]^2 - ξ[j]^2
            end
        end
        if j != N
            η[j] /= ξ[N]^2 - ξ[j]^2
        end
    end

    ξ, η
end

"""
    tabulate(sd::AnalyticalSpectralDensity, full_real::Bool=true)
Returns a table with `ω` and `J(ω)` for ω between -ωmax to ωmax if `full_real` is true. Otherwise the table ranges for ω between 0 and ωmax with `sd.npoints`.
"""
function tabulate(sd::AnalyticalSpectralDensity, full_real::Bool=true)
    ω = Vector{typeof(sd.Δs)}()
    if full_real
        ω = range(-sd.ωmax, sd.ωmax, length=sd.npoints) |> collect
    else
        ωtmp = range(-sd.ωmax, sd.ωmax, length=2 * sd.npoints) |> collect
        ω = ωtmp[sd.npoints+1:end]
    end
    ω, sd.(ω)
end


"""
    SpectralDensityTable <: ContinuousSpectralDensity

Spectral density provided in tabular form. Contains a vector of `ω`s and a vector corresponding to `jw`s.
"""
struct SpectralDensityTable <: ContinuousSpectralDensity
    ω::Vector{Float64}
    jw::Vector{Float64}
    classical::Bool
end
function read_jw(filename; skipstart=0, classical=false)
    w_jw = readdlm(filename; skipstart)
    SpectralDensityTable(w_jw[:, 1], w_jw[:, 2], classical)
end
function read_jw_over_w(filename; skipstart=0, classical=false)
    w_jw = readdlm(filename; skipstart)
    SpectralDensityTable(w_jw[:, 1], w_jw[:, 2] .* w_jw[:, 1], classical)
end

"""
    tabulate(sd::SpectralDensityTable, full_real::Bool=true)
Returns `sd.ω` and `sd.jw`.
"""
function tabulate(sd::SpectralDensityTable, full_real::Bool=true)
    if full_real
        [-reverse(sd.ω); sd.ω], [-reverse(sd.jw); sd.jw]
    else
        sd.ω, sd.jw
    end
end

@doc raw"""
    reorganization_energy(sd::AnalyticalSpectralDensity)
Calculates the reorganization energy corresponding to any analytical spectral density.

``λ = \frac{Δs^2}{2π}\int_{-∞}^∞ \frac{J(ω)}{ω}\,dω``
"""
function reorganization_energy(sd::AnalyticalSpectralDensity)
    ω, jw = tabulate(sd)
    jw ./= ω
    Utilities.trapezoid(ω, jw) / 2π * sd.Δs^2
end

@doc raw"""
    reorganization_energy(sd::SpectralDensityTable)
Calculates the reorganization energy corresponding to any analytical spectral density.

``λ = \frac{1}{2π}\int_{-∞}^∞ \frac{J(ω)}{ω}\,dω``
"""
function reorganization_energy(sd::SpectralDensityTable)
    ω, jw = tabulate(sd)
    jw ./= ω
    Utilities.trapezoid(ω, jw) / 2π
end

@doc raw"""
    reorganization_energy(sd::DiscreteOscillators)
Calculates the reorganization energy corresponding to a bath of discrete oscillators.

``λ = \frac{1}{π}\sum_n \frac{j_n}{ω_n}``
"""
reorganization_energy(sd::DiscreteOscillators) = Utilities.trapezoid(sd.ω, sd.jw ./ sd.ω; discrete=true) / π

@doc raw"""
    mode_specific_reorganization_energy(sd::DiscreteOscillators)
Calculates the array of reorganization energies corresponding to each mode in a bath of discrete oscillators.

``λ_n = \frac{1}{π}\frac{j_n}{ω_n}``
"""
mode_specific_reorganization_energy(sd::DiscreteOscillators) = sd.jw ./ sd.ω ./ π

function polaron_shielding(sd::AnalyticalSpectralDensity, β::Real)
    ω, jw = tabulate(sd, false)
    jw ./= ω.^2 .* tanh(ω * β / 2)
    Utilities.trapezoid(ω, jw) / π * sd.Δs^2
end
polaron_shielding(sd::DiscreteOscillators, β::Real) = Utilities.trapezoid(sd.ω, sd.jw ./ sd.ω^2 .* coth(ω*β/2); discrete=true) / π

@doc raw"""
    discretize(sd::ContinuousSpectralDensity, num_osc::Int)
Discretizes a continuous spectral density into a set of `num_osc` oscillators by assigning equal portions of the total reorganization energy to each oscillator.
"""
function discretize(sd::ContinuousSpectralDensity, num_osc::Int)
    ωtmp, jw = deepcopy(tabulate(sd, false))
    dω = ωtmp[2] - ωtmp[1]
    jw ./= ωtmp
    Δs = (sd isa AnalyticalSpectralDensity) ? sd.Δs : 1
    ω = zeros(length(ωtmp) + 1)
    ω[2:end] .= ωtmp
    int_jw_over_w = cumsum(jw) * dω * Δs^2 / π
    integral_jw_over_w = zeros(length(ωtmp) + 1)
    integral_jw_over_w[2:end] .= int_jw_over_w
    per_mode_λ = integral_jw_over_w[end] / num_osc
    k = 1
    ωs = zeros(num_osc)
    for j in axes(integral_jw_over_w, 1)
        λj = (k - 0.5) * per_mode_λ
        if integral_jw_over_w[j] ≥ λj
            @assert integral_jw_over_w[j-1] < λj "The quadrature grid is too wide."
            # @assert integral_jw_over_w[j-1] < λj "The quadrature grid is too wide. $(integral_jw_over_w[j-1]), $(integral_jw_over_w[j]), $(λj)"
            ratio = (λj - integral_jw_over_w[j-1]) / (integral_jw_over_w[j] - integral_jw_over_w[j-1])
            ωs[k] = ratio * dω + ω[j-1]
            k += 1
        end
        if k>num_osc
            break
        end
    end

    cs = sqrt(2 * per_mode_λ) * ωs / Δs
    ωs, cs
end

end
