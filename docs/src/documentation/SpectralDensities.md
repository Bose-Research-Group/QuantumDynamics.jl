# Spectral Densities

The interaction of a quantum system with a condensed phase environment is often captured through the spectral density. QuantumDynamics has a built-in support for a few of the most common spectral densities and allows for easy incorporation of other spectral densities.

## Spectral Density Hierarchy
```@docs
SpectralDensities
```

The hierarchy of structures are as follows. All spectral densities are subtypes of the SpectralDensity struct:
```@docs
SpectralDensities.SpectralDensity
```
which is split into two types:
```@docs
SpectralDensities.ContinuousSpectralDensity
```
and
```@docs
SpectralDensities.DiscreteOscillators
```
which encodes a bath of discrete oscillators.

The `ContinuousSpectralDensities` struct is further subtyped into:
```@docs
SpectralDensities.SpectralDensityTable
```
for spectral densities obtained in a tabular form either from experiments or from theoretical simulations, and
```@docs
SpectralDensities.AnalyticalSpectralDensity
```
corresponding to the model spectral densities with particular analytical form.

### Analytical Spectral Densities
Currently only two broad classes of analytical spectral densities are covered:

```@docs
SpectralDensities.ExponentialCutoff
```

```@docs
SpectralDensities.DrudeLorentz
```

```@docs
SpectralDensities.UnderdampedBrownian
```

### Exponential Decompositions
HEOM requires exponential decompositions of the bath response function. This information is stored in
```@docs
SpectralDensities.ExponentialDecomposition
```

Currently, Matsubara and Pade schemes for decomposition is supported:
```@docs
SpectralDensities.matsubara_decomposition
```

```@docs
SpectralDensities.pade_decomposition
```

```@docs
SpectralDensities.get_pade_poles_residues
```

## Utilities for spectral densities
All spectral densities can be evaluated on a regular grid of frequencies to give a table of values using
```@docs
SpectralDensities.tabulate
```

For continuous spectral densities, it is often useful to discretize it into a set of independent oscillators. This can be done using
```@docs
SpectralDensities.discretize
```
This is important for applying quantum-classical algorithms like the Ehrenfest limit or the Quantum-Classical Path Integral.

The reorganization energies of spectral densities can be computed using
```@docs
SpectralDensities.reorganization_energy
```

```@docs
SpectralDensities.mode_specific_reorganization_energy
```
