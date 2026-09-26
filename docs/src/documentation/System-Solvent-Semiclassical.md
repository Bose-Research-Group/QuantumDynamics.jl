# Semiclassical Approaches to Reduced Density Matrix Propagation

```@docs
Systems
```

```@docs
Systems.LinearisedSysPhaseSpace
```

```@docs
Systems.PartialLinearisedSysPhaseSpace
```

```@docs
Systems.CompositeSystem
```

The force on the system from the bath is given by:
```@docs
Systems.Fbath!
```

For simulations that use work with the mapping Hamiltonian approach, we need a few other structures:
```@docs
Systems.MappedSystem
```
```@docs
Systems.SpinMappedSystem
```

```@docs
Systems.γ
```

For sampling the system phase space, we use
```@docs
Systems.sample_XP
```

## Partial Linearized Density Matrix
```@docs
PLDM.propagate
```

```@docs
SpinPLDM.propagate
```

## Linearized Semiclassical Methods
```@docs
LSC.propagate
```

```@docs
SpinLSC.propagate
```

```@docs
SpinLSC.sampling_weight
```
```@docs
SpinLSC.sampling_func_mult
```

From the classical trajectories, the reconstruction of the density matrix is done using:
```@docs
SpinLSC.reconstruct_bare_ρ
```

## Spin-Mapping Approaches
There are several mappings that are required for obtaining the quantum mechanical observables:
```@docs
Systems.transform_op
```

```@docs
Systems.SWTransform
```

# Helpers

For efficient simulations, some in-place helper functions are defined to generate the Liouvillian

```@docs
Systems.get_propagator
```

