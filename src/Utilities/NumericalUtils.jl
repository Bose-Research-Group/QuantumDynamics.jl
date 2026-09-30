using LinearAlgebra
using FLoops
using FFTW

"""
    get_BLAS_implementation()
Reports the BLAS implementation under use. The default implementation used by Julia is OpenBlas. MKL is supported through the external package MKL.jl, which needs to be installed and loaded before the loading of QuantumDynamics.jl
"""
get_BLAS_implementation() = BLAS.get_config()

"""
    trapezoid(x, y; discrete::Bool=false, exec=ThreadedEx())
Returns the trapezoidal integration of y with respect to x. If discrete is set to `true`, then returns sum of y. `exec` encodes the execution paradigm and is one of `seq` or `par`.
"""
function trapezoid(x, y; discrete::Bool=false, exec=ThreadedEx())
    if discrete
        @floop exec for val in y
            @reduce ans = zero(eltype(y)) + val
        end
        ans
    else
        len = length(y)
        @floop exec for j = 1:len-1
            @reduce ans = zero(eltype(y)) + (x[j+1] - x[j]) * (y[j+1] + y[j]) / 2
        end
        ans
    end
end

@doc raw"""
    fourier_transform(x::AbstractArray{<:Real}, f::AbstractArray; neg::Bool=true, unitary::Bool=true)
Returns the Fourier transform of `f`:

``F(k) = \mathcal{N}\,\int_{x_\text{min}}^{x_\text{max}} f(x)\,e^{\pm i k x}\,dx``

Arguments:
- `x`: range over which to do the Riemann sum
- `f`: function whose Fourier transform is required
- `neg` [Default: `true`]: pick the negative sign in the exponent if `neg==true`
- `unitary` [Default: `true`]: choose ``\mathcal{N}=\frac{1}{\sqrt{2\pi}}`` if `unitary==true`, or ``\mathcal{N}=1`` otherwise
"""
function fourier_transform(x::AbstractArray{<:Real}, f::AbstractArray; neg::Bool=true, unitary::Bool=true)
    dx = x[2] - x[1]
    k = fftshift(fftfreq(length(x), 1/dx) * 2π) |> collect
    F = neg ? fftshift(dx * fft(f)) .* exp.(-1im * k * x[1]) : fftshift(dx * bfft(f)) .* exp.(1im * k * x[1])
    if unitary
        k, F ./ sqrt(2π)
    else
        k, F
    end
end

@doc raw"""
    inverse_fourier_transform(k::AbstractArray{<:Real}, F::AbstractArray; x0::Float64=0.0, neg::Bool=true, unitary::Bool=true)
Returns the Fourier transform of `f`:

``f(x) = \mathcal{N}\,\int_{k_\text{min}}^{k_\text{max}} F(k)\,e^{\mp i k x}\,dk``

Arguments:
- `k`: range over which to do the Riemann sum
- `F`: function whose inverse Fourier transform is required
- `x0` [Default: `0.0`]: the first value of the output coordinate
- `neg` [Default: `true`]: pick the positive sign in the exponent if `neg==true`. Notice the reversed convention because this is the inverse Fourier transform.
- `unitary` [Default: `true`]: choose ``\mathcal{N}=\frac{1}{\sqrt{2\pi}}`` if `unitary==true`, or ``\mathcal{N}=\frac{1}{2\pi}`` otherwise
"""
function inverse_fourier_transform(k::AbstractArray{<:Real}, F::AbstractArray; x0::Float64=0.0, neg::Bool=true, unitary::Bool=true)
    dk = k[2]-k[1]
    x = fftshift(fftfreq(length(k), 1/dk) * 2π) |> collect
    f = neg ? dk * bfft(ifftshift(F) .* exp.(1im * ifftshift(k) * x0)) : dk * fft(ifftshift(F) .* exp.(-1im * ifftshift(k) * x0))
    if unitary
        x .- x[1] .+ x0, f ./ sqrt(2π)
    else
        x .- x[1] .+ x0, f ./ (2π)
    end
end

# @doc raw"""
#     fourier_transform(time::AbstractArray{<:Real}, corr::AbstractArray{<:Complex}; full=true, unitary=false)
# Returns the Fourier transform of `corr`:

# ``C(\omega) = \mathcal{N}\,\int_a^\infty C(t)\,e^{i\omega t}\,dt``

# If `unitary` is true, then ``\mathcal{N}=\frac{1}{2\pi}``, else ``\mathcal{N}=1``.

# If `full` is true, then ``a=-\infty``, else ``a=0``.
# """
# function fourier_transform(time::AbstractArray{<:Real}, corr::AbstractArray{<:Complex}; full=true, unitary=false, ωmin::Union{Nothing,Float64}=nothing, ωmax::Union{Nothing,Float64}=nothing, flip_sign=false)
#     dt = time[2] - time[1]
#     ωlim = π / dt
#     dω = π / time[end]
#     ωm = isnothing(ωmin) ? -ωlim : max(-ωlim, ωmin)
#     ωM = isnothing(ωmax) ? ωlim : min(ωlim, ωmax)
#     ω = ωm:dω:ωM |> collect
#     spect = zeros(Complex{real(eltype(corr))}, length(ω))
#     sign = flip_sign ? -1im : 1im
#     if full
#         for (l, w) in enumerate(ω)
#             spect[l] = Utilities.trapezoid(time, corr .* exp.(sign * w * time) + conj.(corr) .* exp.(-sign * w * time))
#         end
#     else
#         for (l, w) in enumerate(ω)
#             spect[l] = Utilities.trapezoid(time, corr .* exp.(sign * w * time))
#         end
#     end

#     spect ./= unitary ? sqrt(2π) : 1

#     ω, spect
# end

# function inverse_fourier_transform(ω::AbstractArray{<:Real}, spect::AbstractArray{<:Complex}; unitary=false)
#     dω = ω[2] - ω[1]
#     dt = π / ω[end]
#     tmax = π / dω
#     time = 0:dt:tmax
#     data = zeros(Complex{real(eltype(spect))}, length(time))
#     for (l, t) in enumerate(time)
#         data[l] = Utilities.trapezoid(ω, spect .* exp.(-1im * ω * t))
#     end

#     data ./= unitary ? sqrt(2π) : 2π
#     time, data
# end


function finite_difference_coeffs(n, order)
    M = zeros(order + 1, order + 1)
    M[1, :] .= 1
    for j = 2:order+1
        for k = 2:order+1
            M[j, k] = (-k + 1)^(j - 1)
        end
    end

    v = zeros(order + 1)
    v[n+1] = 1

    M \ v
end

function aaa(X, Y; tol=1e-10, mmax=60)
    M = length(X); mask = trues(M); R = fill(sum(Y)/M, M)
    xj = Float64[]; yj = Float64[]; C = zeros(M, 0); w = Float64[]
    for m in 1:mmax
        j = argmax(abs.(Y .- R) .* mask)
        push!(xj, X[j]); push!(yj, Y[j]); mask[j] = false
        C = hcat(C, 1 ./ (X .- X[j]))
        A = Y[mask] .* C[mask, :] .- C[mask, :] .* yj'
        w = svd(A).V[:, end]
        N = C * (w .* yj); D = C * w
        R = copy(Y); R[mask] = N[mask] ./ D[mask]
        maximum(abs.(Y .- R)) <= tol * maximum(abs.(Y)) && break
    end
    m = length(xj)
    E = [0 w'; ones(m) Diagonal(xj)]
    B = Diagonal([0; ones(m)])
    λ = eigvals(E, B)
    xj, yj, w, filter(isfinite, λ)   # poles x_k
end

function aaa_poles_pruned(x, y; tol=1e-6, rtol=1e-6, mmax=60)
    _, _, _, xk = aaa(x, y; tol, mmax)
    Φ = 1 ./ (x .- transpose(xk))
    r = Φ \ complex.(y)
    keep = abs.(r) .> rtol * maximum(abs.(r))
    xk_clean = xk[keep]
    Φ2 = 1 ./ (x .- transpose(xk_clean))
    r2 = Φ2 \ complex.(y)
    xk_clean, r2
end
