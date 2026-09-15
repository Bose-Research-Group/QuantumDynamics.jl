using Test
using QuantumDynamics
using LinearAlgebra

function setup_spinboson(ϵ, Δ)
    Hamiltonian = Utilities.create_tls_hamiltonian(; ϵ, Δ)
    Jw = SpectralDensities.SpectralDensity[SpectralDensities.DrudeLorentz(; λ=1.5, γ=7.5, ωmax=100000, npoints=10000000)]
    dt = 0.125
    ntimes = 200
    ρ0 = [1.0+0.0im 0; 0 0]
    β = 0.5
    svec = [1.0 -1.0]
    scolvec = [1.0, -1.0]
    times_HEOM, ρs_HEOM = HEOM.propagate(; Hamiltonian, ρ0, β, dt, ntimes, Jw, sys_ops=[complex.(diagm(scolvec))], num_modes=4, Lmax=4, decomposition="pade", verbose=true)
    (; Hamiltonian, Jw, dt, ntimes, β, svec, ρ0, times_HEOM, ρs_HEOM)
end

@testset "Symmetric Spin-Boson" begin
    spinboson = setup_spinboson(0.0, 2.0)
    @testset "QuAPI" begin
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        t, ρs = QuAPI.propagate(; fbU=fbU, Jw=spinboson.Jw, β=spinboson.β, ρ0=spinboson.ρ0, dt=spinboson.dt, ntimes=spinboson.ntimes, kmax=7, verbose=true)
        display([ρs[:,1,1] spinboson.ρs_HEOM[:,1,1]][1:10, :])
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
    @testset "TEMPO" begin
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        t, ρs = TEMPO.propagate(; fbU=fbU, Jw=spinboson.Jw, β=spinboson.β, ρ0=spinboson.ρ0, dt=spinboson.dt, ntimes=spinboson.ntimes, kmax=7, verbose=true)
        display([ρs[:,1,1] spinboson.ρs_HEOM[:,1,1]][1:10, :])
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
end

@testset "Symmetric Spin-Boson" begin
    spinboson = setup_spinboson(1.0, 2.0)
    @testset "QuAPI" begin
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        t, ρs = QuAPI.propagate(; fbU=fbU, Jw=spinboson.Jw, β=spinboson.β, ρ0=spinboson.ρ0, dt=spinboson.dt, ntimes=spinboson.ntimes, kmax=7, verbose=true)
        display([ρs[:,1,1] spinboson.ρs_HEOM[:,1,1]][1:10, :])
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
    @testset "TEMPO" begin
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        t, ρs = TEMPO.propagate(; fbU=fbU, Jw=spinboson.Jw, β=spinboson.β, ρ0=spinboson.ρ0, dt=spinboson.dt, ntimes=spinboson.ntimes, kmax=7, verbose=true)
        display([ρs[:,1,1] spinboson.ρs_HEOM[:,1,1]][1:10, :])
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
end
