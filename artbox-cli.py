#!/usr/bin/env python3
# Copyright 2018 Stefan Kroboth
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
"""Command Line interface

Author: Stefan Kroboth <stefan.kroboth@gmail.com>
"""

# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=no-member
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
from __future__ import print_function
import os
from artbox.parser import args, parser
from time import time
os.environ['CUDA_DEVICE'] = str(args.gpu)
import pycuda.autoinit
from artbox.reconfile import load_matlab_dataset
from artbox.operators import Operator
from artbox.cg import CG
from artbox.tgv import tgv
from artbox.tools import create_dir, gpu_info


def main():
    """Make pylint happy.
    """
    # set environment variable to use chosen GPU device
    # this needs to be done *before* any GPU stuff is done
    #  os.environ['CUDA_DEVICE'] = str(args.gpu)

    # Print GPU information
    if args.gpu_info:
        gpu_info()

    # if time_iters is chosen, it is not useful to show the progress bar
    if args.time_iters:
        args.no_progress = True

    out = args.out

    # Check if all provided files exist
    for dfile in args.data:
        if not os.path.isfile(dfile):
            parser.error("No file " + dfile + ": exiting.")

###############################################################################
# ITERATE OVER ALL FILES                                                      #
###############################################################################
    for dfile in args.data:
        if args.verbose:
            print("Processing file " + dfile + " ...")

        # create directory based on filename
        if dfile[-4:] == '.mat':
            out_dir = out + "/" + os.path.basename(dfile)[:-4]
        else:
            out_dir = out + "/" + dfile

###############################################################################
# APPLY FORWARD MODEL                                                         #
###############################################################################
        if args.forward:
            if args.verbose:
                print("Applying forward model to image")

            cg_out_dir = out_dir + "/forward"
            create_dir(cg_out_dir, args.y)

            if args.verbose:
                print("Loading Data...")

            data = load_matlab_dataset(dfile, double=args.double)

            if args.verbose:
                print("Building Operator...")

            if args.time_operator:
                operator_time = time()

            op = Operator(data,
                          double=args.double,
                          max_threads=args.max_threads,
                          norm_div=args.norm_div,
                          divide=args.divide,
                          hop=args.hop,
                          divide_adjoint=args.divide_adjoint,
                          divide_forward=args.divide_forward,
                          hop_adjoint=args.hop_adjoint,
                          hop_forward=args.hop_forward,
                          show_kernel_params=args.show_kernel_params,
                          verbose=args.verbose)

            if args.time_operator:
                print("  Operator build time: " + str(time() - operator_time) +
                      " seconds")

            if args.time:
                start_time = time()

            import pycuda.gpuarray as gpuarray
            import numpy as np
            from tools import save_matlab
            result = gpuarray.zeros([data.nT, data.nC], np.complex64,
                                    order='F')
            op.apply(gpuarray.to_gpu(data.object.astype(np.complex64)), result)

            if args.time:
                print("Runtime for file " + dfile + ": " +
                      str(time() - start_time) + " seconds")

            save_matlab(result.get(), out_dir, "result")

###############################################################################
# CG RECONSTRUCTION                                                           #
###############################################################################
        if args.cg:
            if args.verbose:
                print("CG Reconstruction")

            cg_out_dir = out_dir + "/cg"
            create_dir(cg_out_dir, args.y)

            if args.verbose:
                print("Loading Data...")

            loading_time = time()
            data = load_matlab_dataset(dfile, double=args.double)
            print("Loading time: " + str(time() - loading_time) + " seconds")

            if args.time_operator:
                operator_time = time()

            if args.verbose:
                print("Building Operator...")

            op = Operator(data,
                          double=args.double,
                          max_threads=args.max_threads,
                          norm_div=args.norm_div,
                          divide=args.divide,
                          hop=args.hop,
                          divide_adjoint=args.divide_adjoint,
                          divide_forward=args.divide_forward,
                          hop_adjoint=args.hop_adjoint,
                          hop_forward=args.hop_forward,
                          show_kernel_params=args.show_kernel_params,
                          verbose=args.verbose)

            if args.time_operator:
                print("  Operator build time: " + str(time() - operator_time) +
                      " seconds")

            if args.time:
                start_time = time()

            cg = CG(op,
                    data,
                    cg_out_dir,
                    double=args.double,
                    relative_tolerance=args.relative_tolerance,
                    absolute_tolerance=args.absolute_tolerance,
                    iters=args.iters,
                    no_progress=args.no_progress,
                    time_iters=args.time_iters,
                    save_images=args.save_images,
                    save_mat=args.save_matlab,
                    image_format=args.image_format,
                    verbose=args.verbose)

            (_, tot_iters) = cg.run()

            if args.time:
                print("Runtime for file " + dfile + " with " + str(tot_iters) +
                      " iterations: " + str(time() - start_time) + " seconds")


###############################################################################
# TGV RECONSTRUCTION                                                          #
###############################################################################
        if args.tgv:
            if args.verbose:
                print("TGV Reconstruction")

            tgv_out_dir = out_dir + "/tgv"
            create_dir(tgv_out_dir, args.y)

            if args.verbose:
                print("Loading Data...")

            data = load_matlab_dataset(dfile, double=args.double)

            if args.time_operator:
                operator_time = time()

            if args.verbose:
                print("Building Operator...")

            op = Operator(data,
                          double=args.double,
                          max_threads=args.max_threads,
                          norm_div=args.norm_div,
                          divide=args.divide,
                          hop=args.hop,
                          divide_adjoint=args.divide_adjoint,
                          divide_forward=args.divide_forward,
                          hop_adjoint=args.hop_adjoint,
                          hop_forward=args.hop_forward,
                          show_kernel_params=args.show_kernel_params,
                          verbose=args.verbose)

            if args.time_operator:
                print("  Operator build time: " + str(time() - operator_time) +
                      " seconds")

            if args.time:
                start_time = time()

            tot_iters = tgv(op,
                            tgv_out_dir,
                            alpha=args.alpha,
                            tau_p=args.tau_p,
                            tau_d=args.tau_d,
                            reduction=args.reduction,
                            fac=args.fac,
                            iters=args.iters,
                            relative_tolerance=args.relative_tolerance,
                            absolute_tolerance=args.absolute_tolerance,
                            cg=False,
                            inner_iters=args.inner_iters,
                            norm_est=args.norm_est,
                            norm_est_iters=args.norm_est_iters,
                            time_iters=args.time_iters,
                            no_progress=args.no_progress,
                            save_images=args.save_images,
                            save_mat=args.save_matlab,
                            image_format=args.image_format,
                            verbose=args.verbose)

            if args.time:
                print("Runtime for file " + dfile + " with " + str(tot_iters) +
                      " iterations: " + str(time() - start_time) + " seconds")

###############################################################################
# TGV CG RECONSTRUCTION                                                       #
###############################################################################
        if args.tgvcg:
            if args.verbose:
                print("TGV CG Reconstruction")

            tgvcg_out_dir = out_dir + "/tgvcg"
            create_dir(tgvcg_out_dir, args.y)

            if args.verbose:
                print("Loading Data...")

            data = load_matlab_dataset(dfile, double=args.double)

            if args.time_operator:
                operator_time = time()

            if args.verbose:
                print("Building Operator...")

            op = Operator(data,
                          double=args.double,
                          max_threads=args.max_threads,
                          norm_div=args.norm_div,
                          divide=args.divide,
                          hop=args.hop,
                          divide_adjoint=args.divide_adjoint,
                          divide_forward=args.divide_forward,
                          hop_adjoint=args.hop_adjoint,
                          hop_forward=args.hop_forward,
                          show_kernel_params=args.show_kernel_params,
                          verbose=args.verbose)

            if args.time_operator:
                print("  Operator build time: " + str(time() - operator_time) +
                      " seconds")

            if args.time:
                start_time = time()

            tot_iters = tgv(op,
                            tgvcg_out_dir,
                            alpha=args.alpha,
                            tau_p=args.tau_p,
                            tau_d=args.tau_d,
                            reduction=args.reduction,
                            fac=args.fac,
                            iters=args.iters,
                            relative_tolerance=args.relative_tolerance,
                            absolute_tolerance=args.absolute_tolerance,
                            cg=True,
                            inner_iters=args.inner_iters,
                            norm_est=args.norm_est,
                            norm_est_iters=args.norm_est_iters,
                            time_iters=args.time_iters,
                            no_progress=args.no_progress,
                            save_images=args.save_images,
                            save_mat=args.save_matlab,
                            image_format=args.image_format,
                            verbose=args.verbose)

            if args.time:
                print("Runtime for file " + dfile + " with " + str(tot_iters) +
                      " iterations: " + str(time() - start_time) + " seconds")


###############################################################################
# TEST ADJOINT OPERATOR (ENCODING MATRIX)                                     #
###############################################################################
        if args.test_adjoint_encoding_mat:
            if args.verbose:
                print("Test Adjoint Operator (Encoding Matrix)")

            if args.verbose:
                print("Loading Data...")

            data = load_matlab_dataset(dfile, double=args.double)

            if args.time_operator:
                operator_time = time()

            if args.verbose:
                print("Building Operator...")

            op = Operator(data,
                          double=args.double,
                          max_threads=args.max_threads,
                          norm_div=args.norm_div,
                          divide=args.divide,
                          hop=args.hop,
                          divide_adjoint=args.divide_adjoint,
                          divide_forward=args.divide_forward,
                          hop_adjoint=args.hop_adjoint,
                          hop_forward=args.hop_forward,
                          show_kernel_params=args.show_kernel_params,
                          verbose=args.verbose)

            if args.time_operator:
                print("  Operator build time: " + str(time() - operator_time) +
                      " seconds")

            if args.time:
                start_time = time()

            tot_iters = op.test_adjoint(args.iters)

            if args.time:
                print("Runtime for file " + dfile + " with " + str(tot_iters) +
                      " iterations: " + str(time() - start_time) + " seconds")


if __name__ == "__main__":
    main()
