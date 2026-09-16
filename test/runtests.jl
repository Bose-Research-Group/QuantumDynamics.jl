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
    kmax = 7
    times_HEOM, ρs_HEOM = HEOM.propagate(; Hamiltonian, ρ0, β, dt, ntimes, Jw, sys_ops=[complex.(diagm(scolvec))], num_modes=4, Lmax=4, decomposition="pade", verbose=false)
    (; Hamiltonian, Jw, dt, ntimes, β, svec, ρ0, times_HEOM, ρs_HEOM, kmax)
end

@testset "Symmetric Spin-Boson" begin
    spinboson = setup_spinboson(0.0, 2.0)
    @testset "QuAPI" begin
        @info "Test Symmetric Spin-Boson QuAPI"
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        t, ρs = QuAPI.propagate(; fbU=fbU, Jw=spinboson.Jw, β=spinboson.β, ρ0=spinboson.ρ0, dt=spinboson.dt, ntimes=spinboson.ntimes, kmax=spinboson.kmax, extraargs=QuAPI.QuAPIArgs(; cutoff=1e-8), verbose=false)
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
    @testset "TEMPO" begin
        @info "Test Symmetric Spin-Boson TEMPO"
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        t, ρs = TEMPO.propagate(; fbU=fbU, Jw=spinboson.Jw, β=spinboson.β, ρ0=spinboson.ρ0, dt=spinboson.dt, ntimes=spinboson.ntimes, kmax=spinboson.kmax, verbose=false)
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
    @testset "Adaptive Kink QuAPI" begin
        @info "Test Symmetric Spin-Boson Adaptive Kink QuAPI with TTM"
        U = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes, forward_backward=false)
        Us = QuAPI.build_augmented_propagator_kink(; fbU=U, Jw=spinboson.Jw, β=spinboson.β, dt=spinboson.dt, ntimes=spinboson.kmax, extraargs=QuAPI.QuAPIArgs(; cutoff=1e-10, prop_cutoff=1e-8), verbose=false)
        Ts = TTM.get_Ts(Us)
        U0es = TTM.get_propagators_from_Ts(Ts, spinboson.ntimes)
        t, ρs = Utilities.apply_propagator(; propagators=U0es, ρ0=spinboson.ρ0, ntimes=spinboson.ntimes, dt=spinboson.dt)
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
    @testset "PCTNPI" begin
        @info "Test Symmetric Spin-Boson PCTNPI with TTM"
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        Us = PCTNPI.build_augmented_propagator(; fbU, Jw=spinboson.Jw, β=spinboson.β, dt=spinboson.dt, ntimes=spinboson.kmax, verbose=false)
        Ts = TTM.get_Ts(Us)
        U0es = TTM.get_propagators_from_Ts(Ts, spinboson.ntimes)
        t, ρs = Utilities.apply_propagator(; propagators=U0es, ρ0=spinboson.ρ0, ntimes=spinboson.ntimes, dt=spinboson.dt)
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
end

@testset "Asymmetric Spin-Boson" begin
    spinboson = setup_spinboson(1.0, 2.0)
    @testset "QuAPI" begin
        @info "Test Asymmetric Spin-Boson QuAPI"
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        t, ρs = QuAPI.propagate(; fbU=fbU, Jw=spinboson.Jw, β=spinboson.β, ρ0=spinboson.ρ0, dt=spinboson.dt, ntimes=spinboson.ntimes, kmax=spinboson.kmax, extraargs=QuAPI.QuAPIArgs(; cutoff=1e-8), verbose=false)
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
    @testset "TEMPO" begin
        @info "Test Asymmetric Spin-Boson TEMPO"
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        t, ρs = TEMPO.propagate(; fbU=fbU, Jw=spinboson.Jw, β=spinboson.β, ρ0=spinboson.ρ0, dt=spinboson.dt, ntimes=spinboson.ntimes, kmax=spinboson.kmax, verbose=false)
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
    @testset "Adaptive Kink QuAPI" begin
        @info "Test Asymmetric Spin-Boson Adaptive Kink QuAPI with TTM"
        U = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes, forward_backward=false)
        Us = QuAPI.build_augmented_propagator_kink(; fbU=U, Jw=spinboson.Jw, β=spinboson.β, dt=spinboson.dt, ntimes=spinboson.kmax, extraargs=QuAPI.QuAPIArgs(; cutoff=1e-10, prop_cutoff=1e-8), verbose=false)
        Ts = TTM.get_Ts(Us)
        U0es = TTM.get_propagators_from_Ts(Ts, spinboson.ntimes)
        t, ρs = Utilities.apply_propagator(; propagators=U0es, ρ0=spinboson.ρ0, ntimes=spinboson.ntimes, dt=spinboson.dt)
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
    @testset "PCTNPI" begin
        @info "Test Asymmetric Spin-Boson PCTNPI with TTM"
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=spinboson.Hamiltonian, dt=spinboson.dt, ntimes=spinboson.ntimes)
        Us = PCTNPI.build_augmented_propagator(; fbU, Jw=spinboson.Jw, β=spinboson.β, dt=spinboson.dt, ntimes=spinboson.kmax, verbose=false)
        Ts = TTM.get_Ts(Us)
        U0es = TTM.get_propagators_from_Ts(Ts, spinboson.ntimes)
        t, ρs = Utilities.apply_propagator(; propagators=U0es, ρ0=spinboson.ρ0, ntimes=spinboson.ntimes, dt=spinboson.dt)
        @test all(norm.(ρs[:, 1, 1] .- spinboson.ρs_HEOM[:, 1, 1]) .< 1e-2)
        @test all(norm.(ρs[:, 2, 2] .- spinboson.ρs_HEOM[:, 2, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- spinboson.ρs_HEOM[:, 1, 2]) .< 1e-2)
        @test all(norm.(ρs[:, 1, 2] .- conj.(ρs[:, 2, 1])) .< 1e-5)
    end
end
