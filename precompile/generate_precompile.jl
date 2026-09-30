# using SnoopCompile
# 
# inf_timing = @snoopi_deep include("snoop/quapi.jl")
# ts, pcs = SnoopCompile.parcel(inf_timing)
# SnoopCompile.write("precompile_sources", pcs)

using PrecompileTools: @setup_workload, @compile_workload

@compile_workload begin
    H = Utilities.create_tls_hamiltonian(; ϵ=0.0, Δ=2.0)
    ρ0 = Matrix{ComplexF64}([
        1.0 0.0
        0.0 0.0
    ])

    Jw = Vector{SpectralDensities.SpectralDensity}([SpectralDensities.ExponentialCutoff(; ξ=0.1, ωc=7.5)])
    SpectralDensities.reorganization_energy(Jw[1])

    β = 1.0

    dt = 0.25
    ntimes = 10
    barefbU = Propagators.calculate_bare_propagators(; Hamiltonian=H, dt, ntimes)
    bareU = Propagators.calculate_bare_propagators(; Hamiltonian=H, dt, ntimes, forward_backward=false)

    t, ρs = QuAPI.propagate(; fbU=barefbU, Jw=Jw, β, ρ0, dt, ntimes, kmax=5, svec=[1.0 -1.0])
    t, ρs = TEMPO.propagate(; fbU=barefbU, Jw=Jw, β, ρ0, dt, ntimes, kmax=5, svec=[1.0 -1.0], extraargs=TEMPO.TEMPOArgs(; algorithm="naive"))
    t, ρs = TEMPO.propagate(; fbU=barefbU, Jw=Jw, β, ρ0, dt, ntimes, kmax=5, svec=[1.0 -1.0], extraargs=TEMPO.TEMPOArgs(; algorithm="densitymatrix"))
    t, ρs = TEMPO.propagate(; fbU=barefbU, Jw=Jw, β, ρ0, dt, ntimes, kmax=5, svec=[1.0 -1.0], extraargs=TEMPO.TEMPOArgs(; algorithm="fit"))
    U = PCTNPI.build_augmented_propagator(; fbU=barefbU, Jw=Jw, β, dt, ntimes=5, svec=[1.0 -1.0])
    U = Blip.build_augmented_propagator(; fbU=barefbU, Jw=Jw, β, dt, ntimes=5, svec=[1.0 -1.0], exec=FLoops.ThreadedEx())
    U = QuAPI.build_augmented_propagator(; fbU=barefbU, Jw=Jw, β, dt, ntimes=5, svec=[1.0 -1.0], exec=FLoops.ThreadedEx())

    invcm2au = 4.55633e-6

    Jw = Vector{SpectralDensities.SpectralDensity}([SpectralDensities.DrudeLorentz(; λ=1.5, γ=7.5)])
    sys_ops = [[1.0+0.0im 0.0; 0.0 -1.0]]
    num_modes = 3
    Lmax = 3
    t, ρs = HEOM.propagate(; Hamiltonian=H, ρ0, β, Jw, sys_ops, tol=1e-7, num_modes, Lmax, dt, ntimes, decomposition="matsubara")
    t, ρs = HEOM.propagate(; Hamiltonian=H, ρ0, β, Jw, sys_ops, tol=1e-7, num_modes, Lmax, dt, ntimes, decomposition="pade")


    H = Matrix{ComplexF64}([
        0.0 0.000525
        0.000525 0.0
    ]) * invcm2au

    β = 87.7899537566
    svec = [1.0 -1.0]

    state1 = [1.0 0.0; 0.0 0.0]
    A = 1im * Utilities.commutator(H, state1)
    B = copy(state1)
    idmat = [1.0 0.0; 0.0 1.0]

    Jw = [SpectralDensities.ExponentialCutoff(; ξ=0.1, ωc=500invcm2au)]
    At, avgbond = ComplexTNPI.A_of_t(; Hamiltonian=H, β, t=0.0, N=50, Jw, svec, A=idmat)
    Q = real(tr(At*state1))
    time_xi01, corr_xi01, avg_bond_dim_xi01 = ComplexTNPI.complex_correlation_function(; Hamiltonian=H, β, tfinal=200.0, dt=100.0, N=50, Jw, svec, A, B=[B], Z=Q, verbose=false, extraargs=Utilities.TensorNetworkArgs())

    N = 5
    mstnpi = MSTNPI.Setup(N, 5, "Exciton")

    hops = OpSum()
    for (j, jnext) in zip(mstnpi.hamiltonian_indices, mstnpi.hamiltonian_indices[2:end])
        hops += 0.5, "G->E", j, "E->G", jnext
        hops += 0.5, "E->G", j, "G->E", jnext
    end
    Propagators.calculate_bare_propagators_mpo(; Hamiltonian=hops, dt, ndivs=2, mstnpi, verbose=false)
    Propagators.calculate_bare_propagators_mpo(; Hamiltonian=hops, dt, ndivs=2, mstnpi, verbose=false)
    Propagators.calculate_bare_propagators_mpo(; Hamiltonian=hops, dt, ndivs=5, mstnpi, verbose=false)
    Propagators.calculate_bare_propagators_mpo(; Hamiltonian=hops, dt, ndivs=5, mstnpi, verbose=false)
    Propagators.calculate_bare_propagators_mpo(; Hamiltonian=hops, dt, ndivs=2, ntimes=5, mstnpi, verbose=false)

    fbsites = Utilities.fb_siteinds("Exciton", N; conserve_qns=true);
    hops = OpSum();
    for r = 1:2
        for j = 1:N-r
            hops += 0.5/r, "G->E", j, "E->G", j+r
            hops += 0.5/r, "E->G", j, "G->E", j+r
        end
    end
    L = Utilities.calculate_Liouvillian(hops, fbsites.sites_forward, fbsites.sites_backward, fbsites.fb_combiners);
    nsteps = 100
    dt = 0.25
    mid = N÷2 + 1;
    ψ0 = MPS(fbsites.sites_forward, [n==mid ? "E" : "G" for n=1:N]);
    ρ0 = Utilities.density_matrix_mps(ψ0, fbsites);
    # ρs = Vector{MPS}(undef, ntimes+1);
    # ρs[1] = deepcopy(ρ0);
    # for j = 2:ntimes+1
    #     ρs[j] = tdvp(L, dt, ρs[j-1]; cutoff, maxdim, nsteps=5)
    # end
end
