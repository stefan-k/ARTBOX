# Copyright 2018 Stefan Kroboth
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
"""Convenient helper functions

Author: Stefan Kroboth <stefan.kroboth@gmail.com>
"""
from __future__ import print_function
import sys
import os
import numpy as np
from PIL import Image
from scipy.io import savemat
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.elementwise import ElementwiseKernel

# pylint: disable=bad-builtin
# pylint: disable=no-member


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    Snippet adapted from:
    http://code.activestate.com/recipes/577058-query-yesno/

    Args:
        question (str): Question presented to the user.
        default (str): Default answer; 'yes', 'no' or `None`. The latter
            requires the user to provide an answer.

    Returns:
        str: Either "yes" or "no", depending on the user input.
    """
    valid = {"yes": "yes", "y": "yes", "ye": "yes", "no": "no", "n": "no"}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while 1:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def create_dir(dir_name, skip_question=True, default="no",
               question="Directory already exists and files may be" +
               "overwritten. Proceed?"):
    """Convenience function for creating a directory.

    If skip_question is `False`, the user will be ask to overwrite the
    directory if it already exists.

    Args:
        dir_name (str): Name of directory.
        skip_question (bool): If ``True``, directory will either be overwritten
            or not overwritten without asking the user, depending on the value
            of `default`.
        default (str): Default answer if directory already exists ("yes" or
            "no").
        question (str): Question prompted to the user if directory already
            exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    elif not skip_question and query_yes_no(question, default) == "no":
        print("Abort.")
        exit()


def imwrite(fname, data):
    """Write image to file.

    First scales the image to the interval [0 255].

    Args:
        fname (str): Filename.
        data (ndarray): 2D image array.
    """
    data = data.squeeze()
    if data.max() != 0:
        data = np.clip(255*data/data.max(), 0, 255).astype('uint8')
    else:
        data = np.zeros(data.shape, np.uint8)
    image = Image.fromarray(data)
    image.save(fname)


def save_image(img, out_dir, name, image_format='png'):
    """Save image to file.

    More convenient function than `imwrite` because it is easier to provide the
    directory and the image format.

    Args:
        img (ndarray): 2D image array.
        out_dir (str): Output directory.
        name (str): Filename.
        image_format (str): Image extension/format (default: 'png')
    """
    imwrite(str(out_dir) + "/" + str(name) + "." + image_format, img)


def save_matlab(img, out_dir, name):
    """Save a matrix as .mat file

    The file will be saved in `<out_dir>/mat/img_<name>.mat` for some reason.

    Args:
        img (ndarray): 2D image array.
        out_dir (str): Output directory.
        name (str): Filename.
    """
    create_dir(out_dir + "/mat")
    savemat(out_dir + "/mat/img_" + str(name),
            dict({"img_" + str(name): img}), oned_as='row')


def display(string):
    """Display a string without a trailing newline like the print command does.

    Args:
        string (str): String to print.
    """
    sys.stdout.write(string)
    sys.stdout.flush()


def dotc(x, y=None):
    """Calculate complex dot product
    If y is not provided, <x, x> is calculated instead.

    Args:
        x (ndarray): Vector.
        y (ndarray): Vector.

    Returns:
        ndarray: Complex dot product.
    """
    if y is None:
        y = x
    return float(np.vdot(x, y))
    #  return float(np.dot(x.flatten().real, y.flatten().real)) +\
    #      float(np.dot(x.flatten().imag, y.flatten().imag))


def dual_energy(u, alpha0, show_text, on_gpu=True):
    """Calculate and print dual energy

    Args:
        u (gpuarray): Array.
        alpha0 (float): Regularization parameter.
        show_text (bool): If `True`, prints dual energy.

    Returns:
        float: Dual energy.
    """
    if on_gpu:
        dual_en = (u.get()**2).sum()/2.0
    else:
        dual_en = (u**2).sum()/2.0
    if show_text:
        print("TGV2-L2-2D-PD: alpha0 = " + str(alpha0) + " dual energy = " +
              str(dual_en))
    return dual_en


def next_power_of_2(x):
    """Determine the next larger number which is a power of two.

    Args:
        x (float): Input number.

    Returns:
        float: Next larger number which is a power of two.
    """
    return pow(2.0, np.ceil(np.log2(x))).astype(np.int32)


def enlarge_next_power_of_2(shape):
    """Determine the next larger shape which is a power of two.

    Args:
        shape (array or tuple): Array of dimensions.

    Returns:
        ndarray: Dimensions with are a power of two.
    """
    new_shape = np.array(shape)
    new_shape[:] = pow(2.0, np.ceil(np.log2(new_shape[:]))).astype(np.int32)
    return new_shape


###############################################################################
# FORWARD AND BACKWARD DIFFERENCES                                            #
###############################################################################
def dyp(u):
    """Backward finite differences in y direction.

    Args:
        u (ndarray): 2D input array.

    Returns:
        ndarray: Finite difference.
    """
    u = np.mat(u)
    dy = np.vstack((u[1:, :], u[-1, :])) - u
    return np.array(dy)


def dxp(u):
    """Backward finite differences in x direction.

    Args:
        u (ndarray): 2D input array.

    Returns:
        ndarray: Finite difference.
    """
    u = np.mat(u)
    dx = np.hstack((u[:, 1:], u[:, -1])) - u
    return np.array(dx)


def dym(u):
    """Forward finite differences in y direction.

    Args:
        u (ndarray): 2D input array.

    Returns:
        ndarray: Finite difference.
    """
    u = np.mat(u)
    N = u.shape[1]
    dy = np.vstack((u[0:-1, :], np.zeros((1, N), u.dtype))) - \
        np.vstack((np.zeros((1, N), u.dtype), u[0:-1, :]))
    return np.array(dy)


def dxm(u):
    """Forward finite differences in x direction.

    Args:
        u (ndarray): 2D input array.

    Returns:
        ndarray: Finite difference.
    """
    u = np.mat(u)
    N = u.shape[1]
    dx = np.hstack((u[:, 0:-1], np.zeros((N, 1), u.dtype))) -\
        np.hstack((np.zeros((N, 1), u.dtype), u[:, 0:-1]))
    return np.array(dx)


###############################################################################
#  GPU TOOLS                                                                  #
###############################################################################
def format_tuple(tup, join_char="."):
    """Formats a tuple of Version numbers for printing.

    Example:
    (4, 2, 0) turns into 4.2.0

    Args:
        tup (tuple): Version as tuple.
        join_char (char): Character by which numbers are joined (default: ".")

    Returns:
        str: Joined version number.
    """
    return str(join_char.join(map(str, tup)))


def gpu_info():
    """Show GPU information
    """
    print("CUDA Version: " + format_tuple(cuda.get_version()))
    print("CUDA Driver Version: " + str(cuda.get_driver_version()))
    print("Number of CUDA devices: " + str(cuda.Device.count()))
    for i in range(0, cuda.Device(0).count()):
        print("Device number " + str(i))
        print("  Name of CUDA device: " + str(cuda.Device(i).name()))
        print("  Compute capability: " +
              format_tuple(cuda.Device(i).compute_capability()))
        print("  Total Memory: " +
              str(cuda.Device(i).total_memory()/(1024.0**2)) + " MB")
        print("  Maximum number of threads per block: " +
              str(cuda.Device(i).max_threads_per_block))
        print("  PCI Bus ID: " + str(cuda.Device(i).pci_bus_id()))
        for (k, v) in cuda.Device(i).get_attributes().items():
            print("  " + str(k) + ": " + str(v))


# Definition of a generic blocksize for the TGV update kernels
#: Generic blocksize in x
block_size_x = 16
#: Generic blocksize in y
block_size_y = 16
#: Generic block definition
block = (block_size_x, block_size_y, 1)


def get_grid(u, offset=0):
    """Computes grid size based on block_size_x, block_size_y and the array
    size.

    Args:
        u (ndarray): Input array for which gridsize should be calculated.

    Returns:
        tuple: CUDA grid.
    """
    grid = (int(np.ceil((u.shape[0+offset] + block[0] - 1)/block[0])),
            int(np.ceil((u.shape[1+offset] + block[1] - 1)/block[1])))
    return grid


def gpuarray_copy(u):
    """Copes a gpuarray object.

    Args:
        u (gpuarray): Input array.

    Returns:
        gpuarra: Deep copy of input array.
    """
    v = gpuarray.zeros_like(u)
    v.strides = u.strides
    cuda.memcpy_dtod(v.gpudata, u.gpudata, u.nbytes)
    return v


def dotc_gpu(x, y=None):
    """Calculate complex dot product on GPU.
    If y is not provided, <x, x> is calculated instead.

    Args:
        x (ndarray): Vector.
        y (ndarray): Vector.

    Returns:
        ndarray: Absolute of complex dot product.
    """
    if y is None:
        y = x
    return np.abs(gpuarray.dot(x.ravel(), y.ravel().conj()).get())

add_scaled_vector_func = \
    ElementwiseKernel("pycuda::complex<float> *out, \
                      pycuda::complex<float> *in1, \
                      pycuda::complex<float> scal, \
                      pycuda::complex<float> *in2",
                      "out[i] = in1[i] + scal * in2[i]",
                      "add_scaled_vector",
                      preamble="#include <pycuda-complex.hpp>")


def add_scaled_vector(out, inp1, scal, inp2):
    """Perform the following (single precision) calculation on the GPU:

    ``out = inp1 + scal * inp2``

    Args:
        inp1 (gpuarray): First input array.
        inp2 (gpuarray): Second input array.
        scal (float): Scaling parameter.

    Returns:
        gpuarray: Output array.
    """
    add_scaled_vector_func(out, inp1, np.float32(scal), inp2)


add_scaled_vector_double_func = \
    ElementwiseKernel("pycuda::complex<double> *out, \
                      pycuda::complex<double> *in1, \
                      pycuda::complex<double> scal, \
                      pycuda::complex<double> *in2",
                      "out[i] = in1[i] + scal * in2[i]",
                      "add_scaled_vector_double",
                      preamble="#include <pycuda-complex.hpp>")


def add_scaled_vector_double(out, inp1, scal, inp2):
    """Perform the following (double precision) calculation on the GPU:

    ``out = inp1 + scal * inp2``

    Args:
        inp1 (gpuarray): First input array.
        inp2 (gpuarray): Second input array.
        scal (float): Scaling parameter.

    Returns:
        gpuarray: Output array.
    """
    add_scaled_vector_double_func(out, inp1, np.float64(scal), inp2)


add_scaled_vector_vector_func = \
    ElementwiseKernel("pycuda::complex<float> *out, \
                      pycuda::complex<float> *in1, \
                      pycuda::complex<float> *scal, \
                      pycuda::complex<float> *in2",
                      "out[i] = in1[i] + scal[i] * in2[i]",
                      "add_scaled_vector_vector",
                      preamble="#include <pycuda-complex.hpp>")


def add_scaled_vector_vector(out, inp1, scal, inp2):
    """Perform the following (single precision) calculation on the GPU:

    ``out = inp1 + scal * inp2``

    Args:
        inp1 (gpuarray): First input array.
        inp2 (gpuarray): Second input array.
        scal (gpuarray): Scaling array.

    Returns:
        gpuarray: Output array.
    """
    add_scaled_vector_vector_func(out, inp1, scal, inp2)

add_scaled_vector_vector_double_func = \
    ElementwiseKernel("pycuda::complex<double> *out, \
                      pycuda::complex<double> *in1, \
                      pycuda::complex<double> *scal, \
                      pycuda::complex<double> *in2",
                      "out[i] = in1[i] + scal[i] * in2[i]",
                      "add_scaled_vector_vector_double",
                      preamble="#include <pycuda-complex.hpp>")


def add_scaled_vector_vector_double(out, inp1, scal, inp2):
    """Perform the following (double precision) calculation on the GPU:

    ``out = inp1 + scal * inp2``

    Args:
        inp1 (gpuarray): First input array.
        inp2 (gpuarray): Second input array.
        scal (gpuarray): Scaling array.

    Returns:
        gpuarray: Output array.
    """
    add_scaled_vector_vector_double_func(out, inp1, scal, inp2)

sub_scaled_vector_func = \
    ElementwiseKernel("pycuda::complex<float> *out, \
                      pycuda::complex<float> *in1, \
                      pycuda::complex<float> scal, \
                      pycuda::complex<float> *in2",
                      "out[i] = in1[i] - scal * in2[i]",
                      "sub_scaled_vector",
                      preamble="#include <pycuda-complex.hpp>")


def sub_scaled_vector(out, inp1, scal, inp2):
    """Perform the following (single precision) calculation on the GPU:

    ``out = inp1 - scal * inp2``

    Args:
        inp1 (gpuarray): First input array.
        inp2 (gpuarray): Second input array.
        scal (float): Scaling parameter.

    Returns:
        gpuarray: Output array.
    """
    sub_scaled_vector_func(out, inp1, np.float32(scal), inp2)

sub_scaled_vector_double_func = \
    ElementwiseKernel("pycuda::complex<double> *out, \
                      pycuda::complex<double> *in1, \
                      pycuda::complex<double> scal, \
                      pycuda::complex<double> *in2",
                      "out[i] = in1[i] - scal * in2[i]",
                      "sub_scaled_vector_double",
                      preamble="#include <pycuda-complex.hpp>")


def sub_scaled_vector_double(out, inp1, scal, inp2):
    """Perform the following (double precision) calculation on the GPU:

    ``out = inp1 - scal * inp2``

    Args:
        inp1 (gpuarray): First input array.
        inp2 (gpuarray): Second input array.
        scal (float): Scaling parameter.

    Returns:
        gpuarray: Output array.
    """
    sub_scaled_vector_double_func(out, inp1, np.float64(scal), inp2)

inplace_mul = ElementwiseKernel("pycuda::complex<float> *x, \
                                pycuda::complex<float> *y",
                                "x[i] *= y[i]",
                                "inplace_mul")

sens_mul_func = ElementwiseKernel("pycuda::complex<float> *out, \
                                  pycuda::complex<float> *u, \
                                  pycuda::complex<float> *sens, \
                                  int coil, int n_coils",
                                  "out[i] = sens[n_coils*i + coil] * u[i]",
                                  "sens_mul")


def sens_mul(out, u, sens, coil):
    """Multiply an array with the coil sensitivity

    ``out = u * sens[coil]``

    Args:
        out (gpuarray): Output array.
        u (gpuarray): Input array.
        sens (gpuarray): RF sensitivity maps for all RF coils.
        coil (int): Coil index.
    """
    sens_mul_func(out, u, sens, np.int32(coil), np.int32(sens.shape[2]))

conj_sens_mul_func = \
    ElementwiseKernel("pycuda::complex<float> *u, \
                      pycuda::complex<float> *v, \
                      pycuda::complex<float> *sens, \
                      int coil, int n_coils",
                      "u[i] += conj(sens[n_coils*i + coil]) * v[i]",
                      "conj_sens_mul")


def conj_sens_mul(u, v, sens, coil):
    """Multiply an array ``v`` with the coil sensitivity ``sens`` and add it
    to ``u``

    ``u = u + v * conj(sens[coil])``

    Args:
        u (gpuarray): Array.
        v (gpuarray): Array.
        sens (gpuarray): RF sensitivity maps for all RF coils.
        coil (int): Coil index.
    """
    conj_sens_mul_func(u, v, sens, np.int32(coil), np.int32(sens.shape[2]))

slice_coil_func = ElementwiseKernel("pycuda::complex<float> *out, \
                                    pycuda::complex<float> *in, \
                                    int coil, int dim",
                                    "out[i] = in[dim*i + coil]",
                                    "slice_coil")


def slice_coil(inp, outp=None, coil=0):
    """Returns a slice of a 3D-Array (image stack or coil sensitivity) since
    slicing is not implemented in PyCUDA.

    Args:
        inp (gpuarray): Input array.
        outp (gpuarray): Output slice (optional, if not provided, it will be
            created).
        coil (int): Coil index.

    Returns:
        gpuarray: Output array.
    """
    dim = inp.shape[0]
    n_coils = inp.shape[1]
    if outp is None:
        outp = gpuarray.zeros(dim, inp.dtype)
    slice_coil_func(outp, inp, np.int32(coil), np.int32(n_coils))
    return outp

x_sn_mul_func = \
    ElementwiseKernel("pycuda::complex<float> *x_out, \
                      pycuda::complex<float> *x, \
                      pycuda::complex<float> *sn, \
                      int Nd1, int Kd1",
                      "x_out[i] = x[(i%Nd1)*Kd1+(i/Nd1)] * conj(sn[i])",
                      "x_sn_mul")


def x_sn_mul(x, sn, Nd, Kd):
    """Multiply ``x`` by ``conj(sn)`` while taking care of the dimensions
    (needed for NUFFT package)

    Args:
        x (gpuarray): Input array.
        sn (gpuarray): sn array.
        Nd (int): Dimension.
        Kd (int): Dimension.

    Returns:
        gpuarray: Output array.
    """
    xout = gpuarray.empty((int(Nd[0]), int(Nd[1])), np.complex64)
    x_sn_mul_func(xout, x, sn, np.int32(Nd[0]), np.int32(Kd[0]))
    return xout

arr_pad_func = ElementwiseKernel("pycuda::complex<float> *in, \
                                 pycuda::complex<float> *out, \
                                 int dim_in, int dim_out",
                                 "out[(i%dim_in)+(i/dim_in)*dim_out] = in[i]",
                                 "arr_pad")


def arr_pad(x, dims):
    """Basically zeropadding an array to ``dims`` dimensions.
    Implemented as follows:
    Write a smaller array into a bigger one. The bigger array will be created
    according to ``dims``. The place of the smaller matrix will be in the upper
    left corner of the bigger array.

    Args:
        x (gpuarray): Input array.
        dims (tuple): Dimensions of the bigger array.

    Returns:
        gpuarray: Output array of size `dims` with `x` in the upper left
            corner.
    """
    out = gpuarray.zeros(dims, x.dtype)
    arr_pad_func(x, out, np.int32(x.shape[0]), np.int32(dims[0]))
    return out

fill_v_func = ElementwiseKernel("pycuda::complex<float> *v_in, \
                                pycuda::complex<float> *v_full, \
                                int coil, int dim",
                                "v_full[dim*i + coil] = v_in[i]",
                                "v_fill")


def fill_v(v, vin, coil):
    """Write one slice in to a 3D stack of images.

    Args:
        v (gpuarray): 3D array.
        vin (gpuarray): 2D input array.
        coil (int): RF coil index.
    """
    dim = np.int32(v.shape[1])
    fill_v_func(vin, v, np.int32(coil), dim)
