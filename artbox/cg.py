# Copyright 2018 Stefan Kroboth
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
"""Conjugate Gradient algorithm

Author: Stefan Kroboth <stefan.kroboth@gmail.com>
"""

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=relative-import
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals

from __future__ import print_function
import pycuda.gpuarray as gpuarray
import pycuda.compiler as compiler
from pycuda.elementwise import ElementwiseKernel
from artbox.tools import block_size_x, block_size_y, block, get_grid,\
    gpuarray_copy, add_scaled_vector_vector_double, add_scaled_vector_vector,\
    save_image, save_matlab, add_scaled_vector, add_scaled_vector_double,\
    dotc_gpu, sub_scaled_vector_double, sub_scaled_vector
import numpy as np

KERNELS = """
#include <pycuda-complex.hpp>

typedef pycuda::complex<float> complex_float;
typedef pycuda::complex<double> complex_double;

__global__ void inner_cg_rhs(complex_float *rhs, const complex_float *u,
                             const complex_float *v, const complex_float *EHs,
                             float tau, int nx, int ny,
                             int chans)
{
  int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
  int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

  if ((x < nx) && (y < ny)) {
    complex_float div = EHs[y*nx+x];
    if (x < nx-1) div += v[    y   *nx+x];
    if (x > 0)    div -= v[    y   *nx+x-1];
    if (y < ny-1) div += v[(ny+y  )*nx+x];
    if (y > 0)    div -= v[(ny+y-1)*nx+x];

    rhs[y*nx+x] = u[y*nx+x] + tau*(div);
  }
}
"""
KERNELS = KERNELS % {
    'BLOCK_SIZE_X': block_size_x,
    'BLOCK_SIZE_Y': block_size_y,
    }

MODULE = compiler.SourceModule(KERNELS)
inner_cg_rhs_func = MODULE.get_function("inner_cg_rhs")


def inner_cg_rhs(rhs, u, v, EHs, tau):
    """Compute right hand side for inner CG method

    ``rhs = u^n + tau * (div_h v^{n+1} + EHs)``

    Args:
        rhs (gpuarray): Right hand side.
        u (gpuarray): u.
        v (gpuarray): v.
        EHs (gpuarray): EHs.
        tau (float): tau.
    """
    inner_cg_rhs_func(rhs, u, v, EHs, np.float32(tau), np.int32(u.shape[0]),
                      np.int32(u.shape[1]), np.int32(1), block=block,
                      grid=get_grid(u))

update_m_func = \
    ElementwiseKernel("pycuda::complex<float> *m, \
                      pycuda::complex<float> alpha, \
                      pycuda::complex<float> *p_k",
                      "m[i] += alpha * p_k[i]",
                      "update_m",
                      preamble="#include <pycuda-complex.hpp>")

update_m_double_func = \
    ElementwiseKernel("pycuda::complex<double> *m, \
                      pycuda::complex<double> alpha, \
                      pycuda::complex<double> *p_k",
                      "m[i] += alpha * p_k[i]",
                      "update_double_m",
                      preamble="#include <pycuda-complex.hpp>")


def update_m(m, alpha, p_k):
    """Performs the following (single precision) calculation:

    ``m = m + alpha * p_k``

    Args:
        m (gpuarray): m.
        alpha (float): alpha.
        p_k (gpuarray): p_k.
    """
    update_m_func(m, np.float32(alpha), p_k)


def update_m_double(m, alpha, p_k):
    """Performs the following (double precision) calculation:

    ``m = m + alpha * p_k``

    Args:
        m (gpuarray): m.
        alpha (float): alpha.
        p_k (gpuarray): p_k.
    """
    update_m_double_func(m, np.float64(alpha), p_k)


class CG(object):
    """Conjugate Gradient method

    Args:
        operator (:class:`artbox.operators.Operator`):
            :class:`artbox.operators.Operator` object.
        data (:class:`artbox.reconfile.ReconData`):
            :class:`artbox.reconfile.ReconData` object.
        out_dir (str): Output directory.
        double (bool): Indicate whether computations should be performed with
            double precision.
        relative_tolerance (float): Relative tolerance for early stopping rule.
        absolute_tolerance (float): Absolute tolerance for early stopping rule.
        iters (int): Number of iterations.
        no_progress (bool): If `True`, no progress bar is shown.
        time_iters (bool): If `True`, all iterations are timed.
        save_images (bool): If `True`, all intermediate images are saved to
            disk.
        save_mat (bool): If `True`, all intermediate images are saved as MATLAB
            data files.
        image_format (str): Format images are saved in.
        verbose (int): Verbosity level.
    """
    def __init__(self, operator, data, out_dir="results/cg", double=False,
                 relative_tolerance=1e-20, absolute_tolerance=1e-19, iters=100,
                 no_progress=False, time_iters=False, save_images=False,
                 save_mat=True, image_format='png', verbose=0):
        self._data = data
        self._op = operator
        if data.weights is not None:
            self._weights = gpuarray.to_gpu(data.weights
                                            .astype(
                                                self._op.precision_complex))
        else:
            self._weights = None

        self._iters = iters
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance
        src_shape = (self._data.nX1, self._data.nX2, 1)
        self._dest_shape = (self._data.nT, self._data.nC)
        self.converged = False
        self.iteration = 0
        self._double = double
        self._no_progress = no_progress
        self._time_iters = time_iters
        self._image_format = image_format
        self._save_images = save_images
        self._save_matlab = save_mat
        self._verbose = verbose
        self._noisy = verbose > 3
        self._out_dir = out_dir
        try:
            recondata_gpu = self._op.dgpu['recondata']
        except NameError:
            recondata_gpu = gpuarray.to_gpu(self._data.recondata)

        # y
        EHs = gpuarray.zeros(src_shape, dtype=self._op.precision_complex)
        self._op.adjoint(recondata_gpu, EHs)

        self._v_k = gpuarray.zeros_like(EHs)

        self._residual_k = gpuarray_copy(EHs)
        self._m = gpuarray.zeros(src_shape,
                                 dtype=self._op.precision_complex)
        self._forward(self._m, self._v_k)

        self._residual_k -= self._v_k
        self._v_k = gpuarray_copy(self._residual_k)

        self._rho_0 = measure(self._v_k, self._residual_k)
        self._rho_k = self._rho_0

        if self._rho_0 <= self._absolute_tolerance:
            if self._verbose:
                print("Already converged at initialization step!")
            self.converged = True
            self.iteration = 0

        self._p_k = gpuarray_copy(self._v_k)

    def _forward(self, m, v_k):
        """Forward Operator ``E^H E``

        Args:
            m (gpuarray): Input array.
            v_k (gpuarray): Output array.
        """
        tmp = gpuarray.zeros(self._dest_shape,
                             dtype=self._op.precision_complex)

        self._op.apply(m, tmp)
        self._op.adjoint(tmp, v_k)
        # v_k = v_k + bla.*m
        if self._double:
            add_scaled_vector_vector_double(v_k, v_k, self._weights, m)
        else:
            add_scaled_vector_vector(v_k, v_k, self._weights, m)
        tmp.gpudata.free()

    def run(self):
        """Runs the conjugate gradient method.
        """
        if not self._no_progress and self._verbose:
            from progressbar import ProgressBar
            progress = ProgressBar()
            iter_range = progress(range(self._iters))
        else:
            iter_range = range(self._iters)

        if self._no_progress and self._time_iters:
            from time import time

        i = 0
        try:
            for i in iter_range:
                if self._verbose and self._no_progress:
                    print("Iteration " + repr(i))

                if self._no_progress and self._time_iters:
                    start = time()

                self.iteration += 1

                self._forward(self._p_k, self._v_k)
                sigma_k = measure(self._p_k, self._v_k)
                alpha_k = self._rho_k / sigma_k
                if self._double:
                    update_m_double(self._m, alpha_k, self._p_k)
                    sub_scaled_vector_double(self._residual_k,
                                             self._residual_k,
                                             alpha_k, self._v_k)
                else:
                    update_m(self._m, alpha_k, self._p_k)
                    sub_scaled_vector(self._residual_k, self._residual_k,
                                      alpha_k, self._v_k)
                self._v_k = gpuarray_copy(self._residual_k)
                rho_k_plus_1 = measure(self._v_k, self._residual_k)
                rho_k_t = np.abs(rho_k_plus_1)

                if (rho_k_t / self._rho_0 <= self._relative_tolerance) \
                   or (rho_k_t <= self._absolute_tolerance):
                    print("Converged.")
                    self.converged = True
                    break

                if self._double:
                    add_scaled_vector_double(self._p_k, self._v_k,
                                             rho_k_plus_1/self._rho_k,
                                             self._p_k)
                else:
                    add_scaled_vector(self._p_k, self._v_k,
                                      rho_k_plus_1/self._rho_k, self._p_k)

                self._rho_k = rho_k_plus_1

                if self._noisy:
                    print(" Residual=" + str(rho_k_t))

                if self._no_progress and self._time_iters:
                    print("Elapsed time for iteration " + str(i) + ": " +
                          str(time() - start) + " seconds")

                if self._save_images:
                    save_image(np.abs(self._m.get().reshape(self._data.nX1,
                                                            self._data.nX2)),
                               self._out_dir, i, self._image_format)
                if self._save_matlab:
                    save_matlab(self._m.get().reshape(self._data.nX1,
                                                      self._data.nX2),
                                self._out_dir, i)
        except KeyboardInterrupt:
            print("Reconstruction aborted (CTRL-C) at iteration " + str(i))
        finally:
            save_image(np.abs(self._m.get().reshape(self._data.nX1,
                                                    self._data.nX2)),
                       self._out_dir, "result", self._image_format)
            save_matlab(self._m.get().reshape(self._data.nX1,
                                              self._data.nX2),
                        self._out_dir, "result")
            self.iteration = i+1
        return (self._m.get().reshape(self._data.nX1, self._data.nX2),
                self.iteration)


class InnerCG(object):
    """Class for conjugate gradient method that solves the update step of ``u``
    in the TGV-CG reconstruction.

    Args:
        Operator (:class:`artbox.operators.Operator`):
            :class:`artbox.operators.Operator` object.
        data (:class:`artbox.reconfile.ReconData`):
            :class:`artbox.reconfile.ReconData` object.
        u (gpuarray): u.
        v (gpuarray): v.
        double (bool): Indicate whether computations should be performed with
            double precision. (TODO!)
        tau (float): tau.
        inner_iters (int): Number of iterations.
        relative_tolerance (float): Relative tolerance for early stopping rule.
        absolute_tolerance (float): Absolute tolerance for early stopping rule.
        verbose (int): Verbosity level.
        EHs (gpuarray): Inital vector.
    """
    def __init__(self, operator, data, u, v, tau, inner_iters,
                 relative_tolerance=1e-20,
                 absolute_tolerance=1e-19,
                 verbose=0,
                 EHs=None):
        self._data = data
        self._op = operator
        self._iters = inner_iters
        self._tau = tau
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance
        src_shape = (self._data.nX1, self._data.nX2, 1)
        self._dest_shape = (self._data.nT, self._data.nC)
        self.converged = False
        self.iteration = 0
        self._verbose = (verbose > 1)

        try:
            recondata_gpu = self._op.dgpu['recondata']
        except NameError:
            recondata_gpu = gpuarray.to_gpu(self._data.recondata)

        # y
        if EHs is None:
            self.EHs = gpuarray.zeros(src_shape, dtype=np.complex64)
            self._op.adjoint(recondata_gpu, self.EHs)
        else:
            self.EHs = EHs

        self._m = u

        self.rhs = gpuarray.zeros_like(self.EHs)
        inner_cg_rhs(self.rhs, u, v, self.EHs, self._tau)

        self._p_k = gpuarray.zeros_like(self.EHs)
        self._v_k = gpuarray.zeros_like(self.EHs)

        self._residual_k = gpuarray_copy(self.rhs)
        self._forward(self._m, self._v_k)  # initial guess

        self._residual_k = self._residual_k - self._v_k
        self._v_k = gpuarray_copy(self._residual_k)

        self._rho_0 = measure(self._v_k, self._residual_k)
        self._rho_k = self._rho_0

        if self._rho_0 <= self._absolute_tolerance:
            if self._verbose:
                print("Already converged!")
            self.converged = True
            self.iteration = 0

        self._p_k = gpuarray_copy(self._v_k)

    def _forward(self, m, v_k):
        """Forward Operator ``E^H E``

        Args:
            m (gpuarray): Input array.
            v_k (gpuarray): Output array.
        """
        tmp = gpuarray.zeros(self._dest_shape, dtype=np.complex64, order='C')
        self._op.apply(m, tmp)
        self._op.adjoint(tmp, v_k)
        add_scaled_vector(v_k, m, self._tau, v_k)

    def run(self):
        """Runs the conjugate gradient method.
        """
        i = 0
        try:
            for i in range(0, self._iters):
                if self._verbose:
                    print("  Inner CG Iteration " + repr(i))

                self._forward(self._p_k, self._v_k)
                sigma_k = measure(self._p_k, self._v_k)
                alpha_k = self._rho_k / sigma_k
                update_m(self._m, alpha_k, self._p_k)
                sub_scaled_vector(self._residual_k, self._residual_k, alpha_k,
                                  self._v_k)
                self._v_k = gpuarray_copy(self._residual_k)
                rho_k_plus_1 = measure(self._v_k, self._residual_k)
                rho_k_t = np.abs(rho_k_plus_1)

                if (rho_k_t / self._rho_0 <= self._relative_tolerance) \
                   or (rho_k_t <= self._absolute_tolerance):
                    if self._verbose:
                        print("Converged at Iteration " + str(i) + ".")
                    self.converged = True
                    self.iteration = i+1
                    return

                add_scaled_vector(self._p_k, self._v_k,
                                  rho_k_plus_1/self._rho_k,
                                  self._p_k)
                self._rho_k = rho_k_plus_1

                if self._verbose >= 3:
                    print("   Residual=" + repr(rho_k_t))
        except KeyboardInterrupt:
            raise
        finally:
            self.iteration = i+1


def measure(x, y):
    """Compute dot product of ``x`` and ``y``.

    Args:
        x (gpuarray): x.
        y (gpuarray): y.
    """
    return dotc_gpu(x, y)
