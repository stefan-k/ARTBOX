# ARTBOX: Accelerated Reconstruction Toolbox

## Introduction

This piece of software is for *research use only* and *not for diagnostic use*!

ARTBOX is a fast image reconstruction and simulation toolbox for magnetic resonance imaging (MRI).
The operator implemented in this work allows you to incorporate B0 inhomogeneities, non-Cartesian trajectories and even non-linear encoding fields into MR image reconstruction/simulation. 
It furthermore considers the reconstructed pixels to be box functions instead of Diracs. This makes it possible to model intra-voxel dephasing which may improve the conditioning of the inverse.

Several image reconstruction methods (CG, TGV-regularization) are implemented.

Note: Please be aware that parts of this software may be refactored and therefore the API may change. Furthermore the documentation and examples are work in progress. Please report any problems you encounter ([Github Issues.](https://github.com/stefan-k/ARTBOX/issues)).

Features:

* Conjugate gradients solver.
* Chambolle-Pock solver for Total Generalized Variation (TGV) regularized problems. (Based on code by [Kristian Bredies](https://imsc.uni-graz.at/bredies/index.html).)
* Modified Chambolle-Pock solver for TGV regularized problems called TGV-CG.
* Uses the full encoding matrix as forward model. Therefore it is easy to include B0 inhomogeneities, non-Cartesian trajectories and nonlinear SEMs (spatially encoding magnetic fields, also called gradients).
* All operations heavily accelerated with GPU code using pycuda. 
* Modular design. Use the accelerated operators to implement your own algorithms (and ideally issue a pull request).

For detailed information about usage and the API, please see the [Documentation](https://stefan-k.github.io/ARTBOX/)

## Installation

### Requirements

Python modules:
* scipy
* numpy
* progressbar
* Pillow
* pycuda
* argparse
* Sphinx (optional, only to build docs)

### Setup

To install ARTBOX, simply clone the repository and `cd` into the directory.

```console
$ git clone git@github.com:stefan-k/ARTBOX.git
$ cd ARTBOX
```

## Usage

For detailed information take a look at the help message:

```console
$ artbox-cli.py --help
```

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

## Related software

* [AGILE](https://github.com/IMTtugraz/AGILE)
* [AVIONIC](https://github.com/IMTtugraz/AVIONIC)
* [bart](https://mrirecon.github.io/bart/)
* [Gadgetron](https://gadgetron.github.io)
* [gpuNUFFT](https://github.com/andyschwarzl/gpuNUFFT)
* [MIRT](https://web.eecs.umich.edu/~fessler/code)
* [Impatient-MRI](https://github.com/JiadingGai/Impatient-MRI)
* ... and probably many more.

## Contact

[Github Issues.](https://github.com/stefan-k/ARTBOX/issues)

## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
