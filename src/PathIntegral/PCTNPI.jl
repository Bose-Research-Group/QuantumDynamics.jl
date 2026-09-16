module PCTNPI

using LinearAlgebra
using HDF5
using ITensors, ITensorMPS
using FLoops
using ..EtaCoefficients, ..Propagators, ..SpectralDensities, ..Blip, ..Utilities, ..TTM

const references = """
- Bose, A. Pairwise Connected Tensor Network Representation of Path Integrals. Physical Review B 2022, 105 (2), 024309. https://doi.org/10.1103/PhysRevB.105.024309."""

const PCTNPIArgs = Utilities.TensorNetworkArgs

function generate_bottom_right_tensor(ηs, sites, blip_sites, state_to_blip, fbU, k::Int, Δs, sbar)
    @inbounds ind1 = sites[k-1]
    @inbounds ind2 = sites[k]
    @inbounds ind3 = blip_sites[k]'
    @inbounds η11 = [η.η00 for η in ηs]
    @inbounds η10 = [η.η0m[1] for η in ηs]
    tens = ITensor(ind1, ind2, ind3)
    for skp = 1:dim(ind1)
        for sk = 1:dim(ind2)
            @inbounds tens[ind1=>skp, ind2=>sk, ind3=>state_to_blip[sk]] = fbU[sk, skp] *
                                                                 exp.(-sum(Δs[:, sk] .* (real(η11) .* Δs[:, sk] + 2im * imag(η11) .* sbar[:, sk]))) *
                                                                 exp.(-sum(Δs[:, sk] .* (real(η10) .* Δs[:, skp] + 2im * imag(η10) .* sbar[:, skp])))
        end
    end
    tens
end

function generate_bottom_left_tensor(ηs, sites, fbU, k::Int, Δs, sbar)
    @inbounds ind1 = sites[k-1]
    @inbounds ind2 = sites[k]
    @inbounds ind3 = sites[k-1]'
    @inbounds η00 = [η.η00 for η in ηs]
    @inbounds η11 = [η.ηmm for η in ηs]
    @inbounds η10 = [η.η0m[1] for η in ηs]
    tens = ITensor(ind1, ind2, ind3)
    for skp = 1:dim(ind1)
        for sk = 1:dim(ind2)
            @inbounds tens[ind1=>skp, ind2=>sk, ind3=>skp] = fbU[sk, skp] *
                                                   exp.(-sum(Δs[:, skp] .* (real(η00) .* Δs[:, skp] + 2im * imag(η00) .* sbar[:, skp]))) *
                                                   exp.(-sum(Δs[:, sk] .* (real(η11) .* Δs[:, sk] + 2im * imag(η11) .* sbar[:, sk]))) *
                                                   exp.(-sum(Δs[:, sk] .* (real(η10) .* Δs[:, skp] + 2im * imag(η10) .* sbar[:, skp])))
        end
    end
    tens
end

function generate_bottom_center_tensor(ηs, sites, blip_sites, state_to_blip, fbU, k::Int, Δs, sbar)
    @inbounds ind1 = sites[k-1]
    @inbounds ind2 = sites[k]
    @inbounds ind3 = sites[k-1]'
    @inbounds ind4 = blip_sites[k]'
    @inbounds η11 = [η.ηmm for η in ηs]
    @inbounds η10 = [η.ηmn[1] for η in ηs]
    tens = ITensor(ind1, ind2, ind3, ind4)
    for skp = 1:dim(ind1)
        for sk = 1:dim(ind2)
            @inbounds tens[ind1=>skp, ind2=>sk, ind3=>skp, ind4=>state_to_blip[sk]] = fbU[sk, skp] *
                                                                            exp.(-sum(Δs[:, sk] .* (real(η11) .* Δs[:, sk] + 2im * imag(η11) .* sbar[:, sk]))) *
                                                                            exp.(-sum(Δs[:, sk] .* (real(η10) .* Δs[:, skp] + 2im * imag(η10) .* sbar[:, skp])))
        end
    end
    tens
end

function generate_center_tensor(ηs, sites, blip_sites, k::Int, kp::Int, group_Δs, Δs, sbar)
    kkp = k - kp
    @inbounds ind1 = prime(sites[kp], kkp - 1)
    @inbounds ind2 = prime(blip_sites[k], kkp - 1)
    @inbounds ind3 = prime(sites[kp], kkp)
    @inbounds ind4 = prime(blip_sites[k], kkp)
    @inbounds η = [η.ηmn[kkp] for η in ηs]
    tens = ITensor(ind1, ind2, ind3, ind4)
    for skp = 1:dim(ind1)
        for sk = 1:dim(ind2)
            @inbounds tens[ind1=>skp, ind2=>sk, ind3=>skp, ind4=>sk] = exp.(-sum(group_Δs[:, sk] .* (real(η) .* Δs[:, skp] + 2im * imag(η) .* sbar[:, skp])))
        end
    end
    tens
end

function generate_left_tensor(ηs, sites, blip_sites, k::Int, kp::Int, group_Δs, Δs, sbar)
    kkp = k - kp
    @inbounds ind1 = prime(sites[kp], kkp - 1)
    @inbounds ind2 = prime(blip_sites[k], kkp - 1)
    @inbounds ind3 = prime(sites[kp], kkp)
    @inbounds η = [η.η0m[kkp] for η in ηs]
    tens = ITensor(ind1, ind2, ind3)
    for skp = 1:dim(ind1)
        for sk = 1:dim(ind2)
            @inbounds tens[ind1=>skp, ind2=>sk, ind3=>skp] = exp.(-sum(group_Δs[:, sk] .* (real(η) .* Δs[:, skp] + 2im * imag(η) .* sbar[:, skp])))
        end
    end
    tens
end

function generate_right_tensor(ηs, sites, blip_sites, k::Int, kp::Int, group_Δs, Δs, sbar)
    kkp = k - kp
    @inbounds ind1 = prime(sites[kp], kkp - 1)
    @inbounds ind2 = prime(blip_sites[k], kkp - 1)
    @inbounds ind3 = prime(blip_sites[k], kkp)
    @inbounds η = [η.η0m[kkp] for η in ηs]
    tens = ITensor(ind1, ind2, ind3)
    for skp = 1:dim(ind1)
        for sk = 1:dim(ind2)
            @inbounds tens[ind1=>skp, ind2=>sk, ind3=>sk] = exp.(-sum(group_Δs[:, sk] .* (real(η) .* Δs[:, skp] + 2im * imag(η) .* sbar[:, skp])))
        end
    end
    tens
end

function generate_top_tensor(ηs, sites, blip_sites, k::Int, kp::Int, group_Δs, Δs, sbar)
    kkp = k - kp
    @inbounds ind1 = prime(sites[kp], kkp - 1)
    @inbounds ind2 = prime(blip_sites[k], kkp - 1)
    @inbounds η = [η.η0e[kkp] for η in ηs]
    tens = ITensor(ind1, ind2)
    for skp = 1:dim(ind1)
        for sk = 1:dim(ind2)
            @inbounds tens[ind1=>skp, ind2=>sk] = exp.(-sum(group_Δs[:, sk] .* (real(η) .* Δs[:, skp] + 2im * imag(η) .* sbar[:, skp])))
        end
    end
    tens
end

"""
    build_augmented_propagator(; fbU::AbstractArray{ComplexF64,3}, Jw::AbstractVector{T}, β::Real, dt::Real, ntimes::Int, kmax::Union{Int,Nothing}=nothing, svec=[1.0 -1.0], reference_prop=false, verbose::Bool=false, extraargs::PCTNPIArgs=PCTNPIArgs(), output::Union{Nothing,HDF5.Group}=nothing, from_TTM::Bool=false, exec=ThreadedEx()) where {T<:SpectralDensities.SpectralDensity}
Builds the propagators, augmented with the influence of the harmonic baths defined by the spectral densities `Jw`,  upto `ntimes` time-steps using the **PCTNPI approach**. The j^th bath, described by `Jw[j]`, interacts with the system through the diagonal operator with the values of `svec[j,:]`.

Relevant references:
$(references)

Arguments:
- `fbU`: time-series of forward-backward propagators
- `Jw`: array of spectral densities
- `svec`: diagonal elements of system operators through which the corresponding baths interact. QuAPI currently only works for baths with diagonal coupling to the system.
- `β`: inverse temperature
- `dt`: time-step for recording the density matrices
- `ntimes`: number of time steps of simulation
- `extraargs`: extra arguments for the PCTNPI algorithm. Contains the `cutoff` threshold for SVD filtration, the maximum bond dimension, `maxdim`, and the `algorithm` of applying an MPO to an MPS.
- `exec`: FLoops.jl execution policy 
"""
function build_augmented_propagator(; fbU::AbstractArray{ComplexF64,3}, Jw::AbstractVector{T}, β::Real, dt::Real, ntimes::Int, kmax::Union{Int,Nothing}=nothing, svec=[1.0 -1.0], reference_prop=false, verbose::Bool=false, extraargs::PCTNPIArgs=PCTNPIArgs(), output::Union{Nothing,HDF5.Group}=nothing, from_TTM::Bool=false, exec=ThreadedEx()) where {T<:SpectralDensities.SpectralDensity}
    @assert length(Jw) == size(svec, 1)
    ηs = [EtaCoefficients.calculate_η(jw; β, dt, kmax=ntimes, imaginary_only=reference_prop) for jw in Jw]
    sdim2 = size(fbU, 2)
    _, state_to_blip, group_Δs, sbar, Δs = Blip.setup_simulation(svec)

    U0e = zeros(ComplexF64, ntimes, sdim2, sdim2)
    T0e = zero(U0e)

    @inbounds begin
        # calculate for the first time step separately, s1->s2
        U0e[1, :, :] = fbU[1, :, :]
        for (bn, η) in enumerate(ηs)
            infl = exp.(-Δs[bn, :] .* (real(η.η00) * Δs[bn, :] + 2im * imag(η.η00) * sbar[bn, :]))
            for t = 1:sdim2
                U0e[1, t, :] .*= infl
                U0e[1, :, t] .*= infl
                infl1 = exp.(-Δs[bn, t] .* (real(η.η0e[1]) * Δs[bn, :] + 2im * imag(η.η0e[1]) * sbar[bn, :]))
                U0e[1, t, :] .*= infl1
            end
        end
        TTM.update_Ts!(T0e, U0e, 1)
        if !isnothing(output)
            if !from_TTM
                Utilities.check_or_insert_value(output, "U0e", U0e)
            end
            Utilities.check_or_insert_value(output, "T0e", T0e)
            output["U0e"][1, :, :] = U0e[1, :, :]
            output["T0e"][1, :, :] = T0e[1, :, :]
            flush(output)
        end

        sites = siteinds(sdim2, ntimes + 1)
        blip_sites = siteinds(size(group_Δs, 2), ntimes + 1; add_tags="blip_sites")
        for j = 2:ntimes
            if verbose
                @info "Step = $(j)"
            end
            tens = Vector{ITensor}(undef, j * (j+1) ÷ 2)
            # calculation for s1->s_{j+1}
            k = 1
            tens[k] = generate_top_tensor(ηs, sites, blip_sites, j + 1, 1, group_Δs, Δs, sbar)
            k += 1
            for l = j:-1:3
                tens[k] = generate_left_tensor(ηs, sites, blip_sites, l, 1, group_Δs, Δs, sbar)
                k += 1
            end
            tens[k] = generate_bottom_left_tensor(ηs, sites, fbU[1, :, :], 2, Δs, sbar)
            k += 1

            for p = 3:j
                tens[k] = generate_bottom_center_tensor(ηs, sites, blip_sites, state_to_blip, fbU[p-1, :, :], p, Δs, sbar)
                k += 1

                for h = 1:j-p
                    tens[k] = generate_center_tensor(ηs, sites, blip_sites, p + h, p - 1, group_Δs, Δs, sbar)
                    k += 1
                end

                tens[k] = generate_right_tensor(ηs, sites, blip_sites, j + 1, p - 1, group_Δs, Δs, sbar)
                k += 1
            end

            tens[k] = generate_bottom_right_tensor(ηs, sites, blip_sites, state_to_blip, fbU[j, :, :], j + 1, Δs, sbar)
            k += 1

            prop = deepcopy(tens[1])
            for tensor in tens[2:end]
                prop *= tensor
            end
            U0e[j, :, :] = Utilities.convert_ITensor_to_matrix(prop, sites[1], sites[j+1])
            TTM.update_Ts!(T0e, U0e, j)
            if !isnothing(output)
                output["U0e"][j, :, :] = U0e[j, :, :]
                output["T0e"][j, :, :] = T0e[j, :, :]
                flush(output)
            end
        end
    end
    U0e
end

end
