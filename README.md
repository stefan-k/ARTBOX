# ARTBOX: Accelerated Reconstruction Toolbox

## Introduction

This piece of software is for *research use only* and *not for diagnostic use*!


## Installation


## Usage

For detailed information take a look at the help message:

$ artbox/artbox.py --help


## Examples

TODO (Reconstructions using CG, TGV, TGV-CG; Simulations)


## Expected Fileformat

A Matlab .mat file with the following struct is expected:

```
S.k: [nF, nT]  # k-space trajectory
S.SEM: [nF, nX1, nX2]  # spatial magnetic encoding fields
S.Cmat: [nC, nX1, nX2]  # RF coil sensitivity maps
S.b0: [nX1, nX2]  # B0 inhomogeneity map
S.ktime: [1, nT]  # "trajectory" corresponding to B0 inhomogeneity
S.Gmat: [nF, nX1, nX2, 3]  # Derivative of SEMs
S.w: [3, 1]  # Weighting factors for dephasing model
S.regularization_weights: [nX1, nX2]  # weights for spatially dependent Tikhonov regularization
```

As well as one of the following two:
```
S.recondata: [nC, nT]  # measured/simulated data
S.object: [nX1, nX2]  # object to be simulated
```

where:

```
nC ... number of RF coils
nF ... number of encoding fields
nX1 ... resolution in image space of dimension 1
nX2 ... resolution in image space of dimension 2
nT ... number of k-space sampling points
```

`recondata` needs to be provided for reconstructions and ```object``` for simulations.

The struct *must* be called `S` and it must be saved in the 'v7' format in Matlab:

```matlab
save('data.mat', 'S', '-v7');
```

## Contact

[Github Issues.](https://github.com/stefan-k/ARTBOX/issues)

## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
