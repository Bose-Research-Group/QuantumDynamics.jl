module SpectralDensityDecompositions

using LinearAlgebra
using ..SpectralDensities, ..Utilities

"""
    ExponentialDecomposition
Contains the data for the representation of a bath correlation function as a sum of exponentials with complex rates. Used for HEOM.

This stems from relating the contour integral for the correlation function to the exponentials of the poles.  Since C(t) is related to e^{-i w t} where w is the pole and it should be decaying, we need to choose the poles with negative imaginary parts, to ensure decaying C(t). The rates then are i * poles with negative imaginary part.

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

decompose(sd::SpectralDensities.DrudeLorentz, tol::Float64=1e-10) = ExponentialDecomposition([sd.γ + 0.0im], [-1im * sd.λ * sd.γ / sd.Δs^2], [1im * sd.λ * sd.γ / sd.Δs^2])
function decompose(sd::SpectralDensities.UnderdampedBrownian, tol::Float64=1e-10)
    Ω = sqrt(complex(sd.ω0^2 - sd.γ^2/4))
    ν = ComplexF64[ sd.γ/2 - 1im*Ω, sd.γ/2 + 1im*Ω ]
    A = sd.λ * sd.ω0^2 / (2 * sd.Δs^2 * Ω)
    c = ComplexF64[ -A, A ]
    # crossed conjugate coefficients
    ctilde = ComplexF64[ conj(c[2]), conj(c[1]) ]
    ExponentialDecomposition(ν, c, ctilde)
end
"""
    decompose(sd::SpectralDensity, tol::Float64=1e-10)

Returns an ExponentialDecomposition of `sd` with an accuracy of `tol`.  For Drude-Lorentz and UnderdampedBrownian baths, it returns the analytic results.
"""
function decompose(sd::T, tol::Float64=1e-10) where T<:SpectralDensities.SpectralDensity
    ω, jw = SpectralDensities.tabulate(sd, false)
    xk, res = Utilities.aaa_poles_pruned(ω.^2, jw ./ ω; tol)
    poles = [imag(s)<0 ? s : -s for s in sqrt.(complex(xk))]
    ν = 1.0im * poles
    c = [-1.0im * r / 2 for r in res]
    ctilde = similar(c)
    n = length(poles)
    for j in 1:n
        # find mirror pole ζ_k = -conj(ζ_j); default to self (k=j) if none matches
        k = findfirst(l -> isapprox(poles[l], -conj(poles[j]); atol=1e-8), 1:n)
        ctilde[j] = conj(c[something(k, j)])
    end
    ExponentialDecomposition(ν, c, ctilde)
end

function thermal_modes(Jw::T, β::Float64, num_modes::Int; scheme="matsubara") where T<:SpectralDensities.SpectralDensity
    if scheme == "matsubara"
        νn = [2π*k/β for k in 1:num_modes]
        κ  = ones(num_modes)
    elseif scheme == "pade"
        η, κ = get_pade_poles_residues(num_modes, Float64)
        νn = η ./ β
    end
    ν = ComplexF64.(νn)
    c = ComplexF64[-1im * κ[k] * (2/β) * Jw(-1im*νn[k]) for k in 1:num_modes]
    ExponentialDecomposition(ν, c, conj.(c))
end

"""Union of SD-pole and thermal modes into one ExponentialDecomposition."""
function combine_decomposition(sd_modes::ExponentialDecomposition, th_modes::ExponentialDecomposition, β::Float64)
    ζ = -1im .* sd_modes.ν                      # recover raw ω-plane poles
    factor = 1 .+ coth.(β .* ζ ./ 2)
    n = length(ζ)
    new_c = sd_modes.c .* factor
    new_ctilde = similar(new_c)
    for j in 1:n
        k = findfirst(l -> isapprox(ζ[l], -conj(ζ[j]); atol=1e-8), 1:n)
        k = something(k, j)
        new_ctilde[j] = conj(new_c[k])
    end
    ExponentialDecomposition(
        vcat(sd_modes.ν, th_modes.ν),
        vcat(new_c, th_modes.c),
        vcat(new_ctilde, th_modes.ctilde)
    )
end

function thermal_decomposition(sd::SpectralDensities.SpectralDensity, tol::Float64, num_modes::Int, scheme::String, β::Float64)
    th_modes = thermal_modes(sd, β, num_modes; scheme)
    sd_modes = decompose(sd, tol)
    combine_decomposition(sd_modes, th_modes, β), real(2im / β * sum(sd_modes.c / sd_modes.ν.^2))
end


imaginary_response_decomposition(sd::SpectralDensities.SpectralDensity, num_modes::Int) = error("Imaginary response decomposition not implemented for $(typeof(sd)).")
matsubara_decomposition(sd::SpectralDensities.SpectralDensity, num_modes::Int, β::AbstractFloat) = error("Matsubara decomposition not implemented for $(typeof(sd)).")
pade_decomposition(sd::SpectralDensities.SpectralDensity, num_modes::Int, β::AbstractFloat) = error("Pade decomposition not implemented for $(typeof(sd)).")

"""
    matsubara_decomposition(sd::DrudeLorentz, num_modes::Int, β::AbstractFloat)

Implements the Matsubara decomposition for the Drude-Lorentz spectral density.
Returns the decay rates, `γ`, and the expansion coefficients, `c`.
"""
function matsubara_decomposition(sd::SpectralDensities.DrudeLorentz, num_modes::Int, β::AbstractFloat)
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
imaginary_response_decomposition(sd::SpectralDensities.DrudeLorentz, num_modes::Int) = ExponentialDecomposition([sd.γ + 0.0im], [-1im * sd.λ * sd.γ / sd.Δs^2], [1im * sd.λ * sd.γ / sd.Δs^2])
function matsubara_decomposition(sd::SpectralDensities.UnderdampedBrownian, num_modes::Int, β::AbstractFloat)
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
function imaginary_response_decomposition(sd::SpectralDensities.UnderdampedBrownian, num_modes::Int)
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
function pade_decomposition(sd::SpectralDensities.DrudeLorentz, num_modes::Int, β::AbstractFloat)
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
function pade_decomposition(sd::SpectralDensities.UnderdampedBrownian, num_modes::Int, β::AbstractFloat)
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

end
