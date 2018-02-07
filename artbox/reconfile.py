# Copyright 2018 Stefan Kroboth
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
"""Handles the data

Author: Stefan Kroboth <stefan.kroboth@gmail.com>
"""
from __future__ import print_function
import os
import scipy.io
import numpy as np

# Why am I even using pylint?
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-few-public-methods
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=too-many-instance-attributes
# Since I have to use exceptions for control flow (ewww), I want to catch any
# exception, therefore pylint shouldn't complain.
# pylint: disable=bare-except


class ReconData(object):
    """Holds all the data needed for reconstructions/simulations

    Args:
        double (bool): Indicates whether computations should be performed in
            single or double precision.
    """
    def __init__(self, double=False):
        if double:
            self.precision_real = np.float64
            self.precision_complex = np.complex128
        else:
            self.precision_real = np.float32
            self.precision_complex = np.complex64

        #: Measured/Simulated data (only if self.shape is not set)
        #: shape = [nC, nT]
        self.recondata = None
        #: Object used for simulation (only if self.recondata is not set)
        #: shape = [nX1, nX2]
        self.object = None
        #: Weights used for spatially dependent Tikhonov regularization
        #: shape = [nX1, nX2]
        self.weights = None
        #: k-space trajectory
        #: shape = [nF, nT]
        self.k = None
        #: Encoding fields
        #: shape = [nF, nX1, nX2]
        self.psi = None
        #: Gradient of encoding fields (used for dephasing model, optional)
        #: shape = [nF, nX1, nX2, 3]
        self.G_mat = None
        #: Weights for dephasing model (optional)
        #: shape = [3, 1]
        self.w = np.array([0, 0, 0], dtype=self.precision_real)
        #: Field inhomogeneity map
        #: shape = [nX1, nX2]
        self.b0 = None
        #: "trajectory" corresponding to B0 map
        #: shape = [1, nT]
        self.ktime = None
        #: RF sensitivity map
        #: shape = [nC, nX1, nX2]
        self.c_mat = None
        #: Number of RF coils
        self.nC = None
        #: Number of k-space sampling points
        self.nT = None
        #: Number of encoding fields
        self.nF = None
        #: Image space resolution in dimension 1
        self.nX1 = None
        #: Image space resolution in dimension 2
        self.nX2 = None
        #: Product of nX1 and nX2
        self.p_res = None
        #: Product of nX1 and nX2 + padding such that p_res32 % 32 == 0
        self.p_res32 = None

    def set_recondata(self, recondata):
        """Set measured/simulated data needed for reconstructions.

        Args:
            recondata (np.ndarray): Measured/simulated data.
                (shape = [nC, nT])
        """
        if recondata is not None:
            self.recondata = recondata.astype(self.precision_complex).flatten()
            #  self.recondata /= np.abs(self.recondata).max()

    def set_object(self, obj):
        """Set object needed for simulations.

        Args:
            obj (np.ndarray): Array used for forward simulations.
                (shape = [nX1, nX2])
        """
        if obj is not None:
            self.object = obj.astype(self.precision_real).flatten()

    def set_weights(self, weights):
        """Set weights for spatially weighted Tikhonov regularization.

        Args:
            weights (np.ndarray): Spatial weighting in image space.
                (shape = [nX1, nX2])
        """
        if weights is not None:
            self.weights = weights.astype(self.precision_real).flatten()

    def set_traj(self, traj):
        """Set trajectory coordinates.

        Args:
            traj (np.ndarray): Trajectory.
                (shape = [nF, nT])
        """
        self.k = traj.astype(self.precision_real).flatten()
        self.nT = int(traj.shape[1])

    def set_SEM(self, SEM):
        """Set encoding fields.

        Args:
            SEM (np.ndarray): Spatial encoding magnetic fields.
                (shape = [nF, nX1, nX2])
        """
        self.psi = SEM.astype(self.precision_real)
        self.psi = np.reshape(self.psi, (self.psi.shape[0], -1)).flatten()
        self.nF = SEM.shape[0]
        self.nX1 = SEM.shape[1]
        self.nX2 = SEM.shape[2]
        self.p_res = self.nX1 * self.nX2
        self.p_res32 = self.p_res
        if self.p_res % 32 != 0:
            self.p_res32 = (int(np.floor(self.p_res/32))+1)*32
        if self.weights is None:
            self.weights = np.zeros((self.nX1, self.nX2),
                                    dtype=self.precision_complex).flatten()

    def set_Gmat(self, Gmat, w):
        """Set G_mat which holds the derivatives of the encoding fields for the
        dephasing model.

        Args:
            Gmat (np.ndarray): Gradients of encoding fields in spatial
                directions x, y and z.
                (shape = [nF, nX1, nX2, 3])
            w (np.ndarray): Corresponding weights.
                (shape = [3, 1])
        """
        if Gmat is not None and w is not None:
            self.G_mat = Gmat.astype(self.precision_real).flatten()
            self.w = w.astype(self.precision_real).flatten()

    def set_b0(self, b0, ktime):
        """Set B0 map (field inhomogeneity)

        Args:
            b0 (np.ndarray): Field inhomogeneity map
                (shape = [nX1, nX2])
            ktime (np.ndarray): Corresponding "trajectory"
                (shape = [1, nT])
        """
        if b0 is not None and ktime is not None:
            self.b0 = b0.astype(self.precision_real)
            self.ktime = ktime.astype(self.precision_real)

    def set_Cmat(self, Cmat):
        """Set RF sensitiviy maps

        Args:
            Cmat (np.ndarray): RF sensitivity maps
                (shape = [nC, nX1, nX2])
        """
        self.nC = Cmat.shape[0]
        self.c_mat = Cmat.astype(self.precision_complex).flatten()

    def save(self, directory, name):
        """Save all data to `directory/name.npz`

        Args:
            directory (str): Directory to save to.
            name (str): Filename.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        fname = directory + "/" + name + ".npz"
        np.savez(fname, traj=self.k, weights=self.weights, SEM=self.psi,
                 Gmat=self.G_mat, w=self.w, b0=self.b0, ktime=self.ktime,
                 Cmat=self.c_mat, a_res=self.a_res, b_res=self.b_res,
                 res0=self.nX1, res1=self.nX2, recondata=self.recondata,
                 object=self.object)


def load_matlab_dataset(filename, double=False):
    """Load a Matlab struct

    The structs name must be `S` and it's format is expected to be:
        S.k: [nF, nT]

        S.SEM: [nF, nX1, nX2]

        S.Cmat: [nC, nX1, nX2]

        S.b0: [nX1, nX2]

        S.ktime: [1, nT]

        S.Gmat: [nF, nX1, nX2, 3]

        S.w: [3, 1]

        S.regularization_weights: [nX1, nX2]

        As well as one of the following two:

        S.recondata: [nC, nT]

        S.object: [nX1, nX2]

    Args:
        filename (str): File to load.
        double (bool): Whether or not everything should be double or single
            precision.
    """
    if not filename.endswith('.mat'):
        raise IOError("File must be .mat")

    data = ReconData(double=double)

    # ['simuStruct']
    mat = scipy.io.loadmat(filename,
                           struct_as_record=False,
                           squeeze_me=True,
                           mat_dtype=False)['S']
    #  mat_dtype=False)['reconStruct']
    # mat = mat[mat.keys()[0]] # get the only key (doesn't require a name)

    # Encoding fields
    try:
        SEM = np.array(mat.SEM_mat)
    except:
        SEM = np.array(mat.SEM)

    data.set_SEM(SEM)

    # Trajectory
    k = np.array(mat.k)
    data.set_traj(k)

    # b0_mat
    try:
        b0_mat_t = np.array(mat.b0map)
        ktime = np.array(mat.ktime)
        if np.prod(b0_mat_t.shape) <= 0:
            print('Inacceptable b0_mat provided.')
            exit()
            b0_mat_t = None
            ktime = None
    except:
        b0_mat_t = None
        ktime = None

    data.set_b0(b0_mat_t, ktime)

    # G_mat
    try:
        G_mat = np.array(mat.G_mat)
        # possibly problematic:
        #  G_mat = G_mat.transpose((2, 3, 0, 1))
        if np.prod(G_mat.shape) <= 3*data.nX1*data.nX2*data.nF:
            print("Inacceptable G_mat provided. Will not use dephasing model.")
            G_mat = None
    except:
        G_mat = None

    try:
        w = np.array(mat.w)
    except:
        if G_mat is not None:
            print("No field w -- assume w = [0, 0, 0]. This essentially " +
                  "does not consider dephasing in the reconstruction/" +
                  "simulation.")
        w = np.array([0, 0, 0])

    data.set_Gmat(G_mat, w)

    # Tikhonov weights
    # take care of this later when it is not provided!
    try:
        weights_mat_t = np.array(mat.regularization_weights)
    except:
        weights_mat_t = np.zeros((data.nX1, data.nX2))

    data.set_weights(weights_mat_t)

    # coil sensitivites
    try:
        c_mat = np.array(mat.b1_mat)
    except:
        c_mat = np.array(mat.Cmat)

    # perform zeropadding
    if data.p_res != data.p_res32:
        c_mat.resize((c_mat.shape[0], data.p_res32))

    data.set_Cmat(c_mat)

    # get recondata in the desired shape
    try:
        # aquired data
        recondata = np.array(mat.recondata)
        recondata /= np.abs(recondata).max()
        data.set_recondata(recondata)
    except:
        print('Seems like you are trying to simulate?')
        try:
            # should this be real? I mean objects are always real...
            obj = np.array(mat.object)
            data.set_object(obj)
        except:
            print('ERROR: Either recondata or object needs to be provided!')
            exit()
    return data


def load_numpy_dataset(filename, double=False):
    """Load a numpy dataset (.npz)

    The following fields must be present and must have the following formats:

        k: [nF, nT]

        SEM: [nF, nX1, nX2]

        Cmat: [nC, nX1, nX2]

        b0: [nX1, nX2]

        ktime: [1, nT]

        Gmat: [nF, nX1, nX2, 3]

        w: [3, 1]

        regularization_weights: [nX1, nX2]

        As well as one of the following two:

        recondata: [nC, nT]

        object: [nX1, nX2]

    Args:
        filename (str): File to load (must end in .npz).
        double (bool): Whether or not everything should be double or single
            precision.
    """
    if not filename.endswith('.npz'):
        raise IOError("File must be .npz")

    data = ReconData(double=double)

    with np.load(filename) as data:
        # SEM
        data.set_SEM(data['SEM'])

        # Trajectory
        data.set_traj(data['traj'])

        # B0 inhomogeneity map
        data.set_b0(data.get('b0'), data.get('ktime'))

        # G_mat and weights
        data.set_Gmat(data.get('Gmat'), data.get('w'))

        # Tikhonov weights
        data.set_weights(data.get('weights'))

        # RF coil sensitivity maps
        data.set_Cmat(data['Cmat'])

        # data
        data.set_recondata(data.get('recondata'))

        # object
        data.set_recondata(data.get('object'))


def load_dataset(filename, double=False):
    """Load a .mat or .npz dataset.

    See docstring of `load_matlab_dataset` and `load_numpy_dataset` for
    detailed information.

    Args:
        filename (str): File to load (must end in .npz or .mat).
        double (bool): Whether or not everything should be double or single
            precision.
    """
    if filename.endswith('.npz'):
        return load_numpy_dataset(filename, double=double)
    elif filename.endswith('.mat'):
        return load_matlab_dataset(filename, double=double)
    else:
        raise IOError("Wrong file format.") 
