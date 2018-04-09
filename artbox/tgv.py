# Copyright 2018 Stefan Kroboth
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
"""GPU Implementation of the TGV solver (primal dual algorithm)

Based on code by Kristian Bredies <kristian.bredies@uni-graz.at>

Author: Stefan Kroboth <stefan.kroboth@gmail.com>
"""

# pylint: disable=relative-import
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=invalid-name
# pylint: disable=no-member

from __future__ import print_function
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
from artbox.tools import save_image, save_matlab, gpuarray_copy
import artbox.tgv_kernels as tgvk


def tgv(op, out_dir, alpha=4e-5, tau_p=0.625, tau_d=0.125, reduction=2**-8,
        fac=2, iters=3000, relative_tolerance=1e-20, absolute_tolerance=1e-19,
        cg=False, inner_iters=20, norm_est=None, norm_est_iters=10,
        time_iters=False, no_progress=False, save_images=False, save_mat=False,
        image_format='png', verbose=False):
    """TGV regularized reconstruction using the Encoding Matrix E.

    Args:
        op (:class:`artbox.operators.Operator`):
            :class:`artbox.operators.Operator` object.
        out_dir (str): Output directory.
        alpha (float): Regularization parameter.
        tau_p (float): tau_p.
        tau_d (float): tau_d.
        reduction (float): Regularization parameter reduction per iteration.
        fac (float): fac.
        iters (int): Number of iterations.
        relative_tolerance (float): Relative tolerance for early stopping rule.
        absolute_tolerance (float): Absolute tolerance for early stopping rule.
        cg (bool): Indicate whether inner CG method should be used (TGV-CG).
        inner_iters (int): Number of iterations for inner CG.
        norm_est (float): Estimated norm of operator. If `None`, it will be
            calculated.
        norm_est_iters (int): Number of iterations for norm estimation.
        time_iters (bool): If `True`, all iterations are timed.
        no_progress (bool): If `True`, no progress bar is shown.
        save_images (bool): If `True`, all intermediate images are saved to
            disk.
        save_mat (bool): If `True`, all intermediate images are saved as MATLAB
            data files.
        image_format (str): Format images are saved in.
        verbose (int): Verbosity level.
        double (bool): Indicate whether computations should be performed with
            double precision. TODO!!
    """
    data = op.data

    alpha = alpha/reduction
    alpha00 = alpha*fac
    alpha10 = alpha
    alpha01 = alpha00*reduction
    alpha11 = alpha10*reduction

    maxiter = iters

    # set up primal variables
    ut = gpuarray.zeros((data.nX1, data.nX2, 1), np.complex64, order='F')
    # 'reference zero'
    tt = gpuarray.zeros((data.nX1, data.nX2, 1), np.float32, order='F')

    op.adjoint(op.dgpu['recondata'], ut)

    # norm estimation
    if norm_est is None:
        # perform norm estimation
        norm_est = op.norm_est(ut, norm_est_iters)
        if verbose:
            print("Norm estimation: " + str(norm_est))
    else:
        # use user-provided norm
        if verbose:
            print("Norm estimation (provided by user): " + str(norm_est))

    ut /= norm_est
    u = gpuarray.maximum(tt, ut.real).astype(np.complex64)
    u_ = gpuarray_copy(u)
    w = gpuarray.zeros((u.shape[0], u.shape[1], 2*u.shape[2]), np.complex64,
                       order='F')
    w_ = gpuarray.zeros((u.shape[0], u.shape[1], 2*u.shape[2]), np.complex64,
                        order='F')

    # set up dual variables
    p = gpuarray.zeros((u.shape[0], u.shape[1], 2*u.shape[2]), np.complex64,
                       order='F')
    q = gpuarray.zeros((u.shape[0], u.shape[1], 3*u.shape[2]), np.complex64,
                       order='F')
    v = gpuarray.zeros_like(op.dgpu['recondata'])

    # set up variables associated with linear transform
    Ku = gpuarray.zeros(op.dgpu['recondata'].shape, np.complex64, order='F')
    Kadjv = gpuarray.zeros((data.nX1, data.nX2, 1), np.complex64,
                           order='F')

    # if args.L2 is None:
    #   # L2 is *not* provided by the user
    #   M = 1
    #   L2 = 0.5*(M*M + 17 + np.sqrt(pow(M, 4.0) - 2*M*M + 33))
    # else:
    #   # L2 is provided by the user
    #   L2 = args.L2

    # this one works for TGV
    # tau_p = 1.0/np.sqrt(L2)
    # tau_d = 1.0/tau_p/L2

    uold = gpuarray.empty_like(u)
    wold = gpuarray.empty_like(w)

    if cg:
        from artbox.cg import InnerCG
        tmp_EHs = None
        tau_p = 1.0/norm_est
        tau_d = 1.0/tau_p/(0.5*(17+np.sqrt(33)))

    if time_iters:
        from time import time

    if not no_progress and verbose:
        # set up progress bar
        from progressbar import ProgressBar
        progress = ProgressBar()
        iter_range = progress(range(maxiter))
    else:
        iter_range = range(maxiter)

    total_iterations = 0

    try:
        for k in iter_range:
            if verbose and no_progress:
                print("Iteration " + repr(k))

            if no_progress and time_iters:
                start = time()

            total_iterations += 1

            alpha0 = np.exp(float(k)/maxiter*np.log(alpha01) +
                            float(maxiter-k)/maxiter*np.log(alpha00))
            alpha1 = np.exp(float(k)/maxiter*np.log(alpha11) +
                            float(maxiter-k)/maxiter*np.log(alpha10))

            # primal update
            cuda.memcpy_dtod(uold.gpudata, u.gpudata, u.nbytes)
            cuda.memcpy_dtod(wold.gpudata, w.gpudata, w.nbytes)

            op.apply(u_, Ku)
            Ku /= norm_est

            tgvk.tgv_update_v(v, Ku, op.dgpu['recondata'], tau_d,
                              lin_constr=(alpha1 < 0))

            # dual update
            tgvk.tgv_update_p(u_, w_, p, tau_d, abs(alpha1))
            tgvk.tgv_update_q(w_, q, tau_d, abs(alpha0))

            op.adjoint(v, Kadjv)
            Kadjv /= norm_est

            # Inner conjugate gradient method
            if cg:
                try:
                    icg = InnerCG(op, data, u, p, tau_p, inner_iters,
                                  relative_tolerance, absolute_tolerance,
                                  verbose, EHs=tmp_EHs)
                    icg.run()
                except:
                    raise
                finally:
                    total_iterations += icg.iteration
                tmp_EHs = icg.EHs
            else:
                tgvk.tgv_update_u(u, p, Kadjv, tau_p)

            tgvk.tgv_update_w(w, p, q, tau_p)

            # extragradient update
            tgvk.tgv_update_u_2(u_, u, uold)
            tgvk.tgv_update_w_2(w_, w, wold)

            # Print time per iteration
            if no_progress and time_iters:
                print("Elapsed time for iteration " + str(k) + ": " +
                      str(time() - start) + " seconds")

            # Save images
            if save_images:
                save_image(np.abs(u.get().reshape(data.nX1, data.nX2)),
                           out_dir, k, image_format)

            # Save matlab files
            if save_mat:
                save_matlab(u.get().reshape(data.nX1, data.nX2),
                            out_dir, k)
    except KeyboardInterrupt:
        print("Reconstruction aborted (CTRL-C) at iteration " +
              str(total_iterations))
    finally:
        # always save final image and Matlab data
        save_image(np.abs(u.get().reshape(data.nX1, data.nX2)),
                   out_dir, "result", image_format)
        save_matlab(u.get().reshape(data.nX1, data.nX2),
                    out_dir, "result")
    return total_iterations
