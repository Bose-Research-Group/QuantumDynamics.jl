module HEOMStructure

using LinearAlgebra: mul!, axpy!
using ..SpectralDensities, ..Solvents, ..Utilities

"""
    get_vecs(len::Int, L::Int)

Get a vector of vectors of length `len`, where the sum is L.
"""
function get_vecs(len::Int, L::Int)
    len == 1 && return [[L]]
    ans = Vector{Vector{typeof(L)}}()
    for j = 0:L
        rest = get_vecs(len - 1, L - j)
        curr = [cat(j, r; dims=1) for r in rest]
        append!(ans, curr)
    end
    ans
end

function get_mode_map(decompositions)
    mode_map = Tuple{Int,Int}[]
    for (b,dec) in enumerate(decompositions)
        for k in eachindex(dec.ν)
            push!(mode_map,(b,k))
        end
    end
    mode_map
end

"""
    setup_simulation(decompositions::Vector{SpectralDensities.ExponentialDecomposition}, Lmax::Int)

Sets up the HEOM hierarchy for a collection of bath exponential decompositions.

Each bath is represented by an `ExponentialDecomposition` containing the
decay constants and coefficients of its exponential expansion. The hierarchy
is constructed over the total number of exponential terms across all baths.

Arguments:
- `decompositions`: Vector of exponential decompositions, one for each bath.
- `Lmax`: Maximum hierarchy depth.

Returns:
- `nveclist`: List of hierarchy index vectors. Each vector contains the
  occupation numbers of all exponential terms across all baths.
- `npluslocs`: `npluslocs[k,l]` gives the location of the hierarchy vector
  obtained by increasing the occupation number of exponential term `k` in the
  `l`-th hierarchy element by one. A value of zero indicates that the resulting
  vector lies outside the truncated hierarchy.
- `nminuslocs`: `nminuslocs[k,l]` gives the location of the hierarchy vector
  obtained by decreasing the occupation number of exponential term `k` in the
  `l`-th hierarchy element by one. A value of zero indicates that the operation
  is not allowed.
- `mode_map`: Mapping from the flattened exponential index to the bath and
  local exponential index. Specifically, `mode_map[k] = (b,m)` means that the
  `k`-th hierarchy mode corresponds to the `m`-th exponential term of bath `b`.

The flattening of exponential modes allows the hierarchy implementation to
handle arbitrary spectral density decompositions, including baths with
different numbers of exponential terms and complex-conjugate pole pairs.
"""
function setup_simulation(decompositions::Vector{SpectralDensities.ExponentialDecomposition}, Lmax::Int)
    # Map flattened exponential index -> (bath index, local exponential index)
    mode_map = get_mode_map(decompositions)
    num_modes = length(mode_map)

    # Generate hierarchy index vectors
    nveclist = Vector{Vector{Int}}()

    for L = 0:Lmax
        append!(nveclist, get_vecs(num_modes, L))
    end

    Nh = length(nveclist)
    # Map hierarchy vectors to their locations
    index = Dict{Tuple{Vararg{Int}},Int}()
    for (i, nvec) in enumerate(nveclist)
        index[Tuple(nvec)] = i
    end

    # Neighbor locations
    npluslocs  = zeros(Int, num_modes, Nh)
    nminuslocs = zeros(Int, num_modes, Nh)

    for (j,nvec) in enumerate(nveclist)
        for k = 1:num_modes
            # Increase occupation number
            nvec_plus = copy(nvec)
            nvec_plus[k] += 1

            npluslocs[k,j] = get(index, Tuple(nvec_plus), 0)

            # Decrease occupation number
            if nvec[k] > 0
                nvec_minus = copy(nvec)
                nvec_minus[k] -= 1

                nminuslocs[k,j] = get(index, Tuple(nvec_minus), 0)
            end
        end
    end

    nveclist, npluslocs, nminuslocs, mode_map
end

function get_decay(nveclist, mode_map, decomps)
    decay = zeros(ComplexF64, length(nveclist))
    for (i, nvec) in enumerate(nveclist)
        val = 0.0 + 0.0im
        for k in eachindex(nvec)
            bath, mode = mode_map[k]
            val += nvec[k] * decomps[bath].ν[mode]
        end
        decay[i] = val
    end
    decay
end

struct HEOMParams{Ltype <: Union{Nothing, Vector{Matrix{ComplexF64}}}, EField <: Union{Nothing, Vector{Utilities.ExternalField}, Tuple{Solvents.Solvent, Solvents.PhaseSpace}}}
    H::Matrix{ComplexF64}
    L::Ltype
    LdagL::Ltype
    external_fields::EField
    coupl::Vector{Matrix{ComplexF64}}
    coupl2::Vector{Matrix{ComplexF64}}
    nveclist::Vector{Vector{Int}}
    npluslocs::Matrix{Int}
    nminuslocs::Matrix{Int}
    mode_map::Vector{Tuple{Int,Int}}
    decomps::Vector{SpectralDensities.ExponentialDecomposition}
    Δk::Vector{Float64}
    β::Float64
    decay::Vector{ComplexF64}
    raising_coeffs::Matrix{Float64}
    lowering_coeffs::Matrix{ComplexF64}
    workspace::Matrix{ComplexF64}
    tmp1::Matrix{ComplexF64}
end

function get_hierarchy_coeffs(nveclist, mode_map, npluslocs, nminuslocs, decomps)
    num_modes = length(mode_map)
    Nh = length(nveclist)
    raising_coeff  = zeros(Float64, num_modes, Nh)
    lowering_coeff = zeros(ComplexF64, num_modes, Nh)
    for n in 1:Nh
        nvec = nveclist[n]
        for k in 1:num_modes
            bath, mode = mode_map[k]
            dec = decomps[bath]
            if npluslocs[k, n] > 0
                raising_coeff[k, n] = sqrt((nvec[k] + 1) * dec.scale[mode])
            end
            if nminuslocs[k, n] > 0
                lowering_coeff[k, n] = -1im * sqrt(nvec[k] / dec.scale[mode])
            end
        end
    end
    raising_coeff, lowering_coeff
end

function get_eff_hamiltonian!(tmp1, H, ext_fields::Nothing, t)
    tmp1 .= H
    nothing
end
function get_eff_hamiltonian!(tmp1, H, ext_fields::Vector{Utilities.ExternalField}, t)
    tmp1 .= H
    for ef in ext_fields
        tmp1 .+= ef.V(t) * ef.coupling_op
    end
    nothing
end
function get_eff_hamiltonian!(tmp1, H, ext_fields::Tuple{Solvents.Solvent, Solvents.PhaseSpace}, t)
    tmp1 .= H + Solvents.get_Vint(ext_fields[2], ext_fields[1], t)
    nothing
end
function get_base_eom!(dρ, ρ, H, tmp1, external_fields, t)
    @inbounds begin
        get_eff_hamiltonian!(tmp1, H, external_fields, t)
        for n in axes(ρ, 3)
            @views begin
                mul!(dρ[:, :, n], tmp1, ρ[:, :, n], -1im, 0.0)
                mul!(dρ[:, :, n], ρ[:, :, n], tmp1', 1im, 1.0)
            end
        end
    end
    
    nothing
end

function uncoupled_eom!(dρ, ρ, params::HEOMParams{Nothing, T}, t) where T
    get_base_eom!(dρ, ρ, params.H, params.tmp1, params.external_fields, t)
    nothing
end
function uncoupled_eom!(dρ, ρ, params::HEOMParams{Vector{Matrix{ComplexF64}}, T}, t) where T
    get_base_eom!(dρ, ρ, params.H, params.tmp1, params.external_fields, t)
    for (L, LdagL) in zip(params.L, params.LdagL)
        Ldag = L'
        for n in axes(ρ, 3)
            ρn = @view ρ[:, :, n]
            dρn = @view dρ[:, :, n]
            mul!(params.tmp1, ρn, Ldag)
            mul!(dρn, L, params.tmp1, 1.0, 1.0)
            mul!(dρn, LdagL, ρn, -0.5, 1.0)
            mul!(dρn, ρn, LdagL, -0.5, 1.0)
            # dρn .+= L * ρn * Ldag .- 0.5 .* LdagL * ρn .- 0.5 .* ρn * LdagL
        end
    end
    nothing
end
function scaled_HEOM_RHS!(dρ, ρ, params, t)
    @inbounds begin
        uncoupled_eom!(dρ, ρ, params, t)
        for n in axes(ρ, 3)
            @views begin
                ρn = ρ[:, :, n]
                dρn = dρ[:, :, n]
                nvec = params.nveclist[n]
                npluslocs = params.npluslocs[:, n]
                nminuslocs = params.nminuslocs[:, n]
                ρplus = params.workspace
            end

            # ADO decay
            @. dρn -= params.decay[n] * ρn

            # Residual correction terms (one per bath)
            for (Δk, co, co2) in zip(params.Δk, params.coupl, params.coupl2)
                Utilities.double_commutator!(params.workspace, co, co2, ρn, params.tmp1)
                dρn .-= Δk .* params.workspace
            end

            # Loop over baths
            for (bath, co) in enumerate(params.coupl)
                dec = params.decomps[bath]
                fill!(ρplus, 0.0)

                # Loop over all exponential modes belonging to this bath
                for k in eachindex(nvec)
                    mode_bath, mode_local = params.mode_map[k]
                    if mode_bath != bath
                        continue
                    end
                    mode = mode_local

                    # Raising contribution
                    loc_plus = npluslocs[k]
                    if loc_plus > 0
                        @views axpy!(params.raising_coeffs[k, n], ρ[:, :, loc_plus], ρplus)
                    end
                    # Lowering contribution
                    loc_minus = nminuslocs[k]
                    if loc_minus > 0
                        @views mul!(dρn, co, ρ[:, :, loc_minus], params.lowering_coeffs[k, n] * dec.c[mode], 1.0)
                        @views mul!(dρn, ρ[:, :, loc_minus], co, -params.lowering_coeffs[k, n] * dec.ctilde[mode], 1.0)
                    end
                end

                # Apply commutator once per bath
                mul!(dρn, co, ρplus, -1.0im, 1.0)
                mul!(dρn, ρplus, co, 1.0im, 1.0)
            end
        end
    end
    nothing
end
end
