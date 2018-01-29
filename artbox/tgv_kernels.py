# Copyright 2018 Stefan Kroboth
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
"""TGV GPU Kernels

Based on code by Kristian Bredies <kristian.bredies@uni-graz.at>

Author: Stefan Kroboth <stefan.kroboth@gmail.com>
"""

# pylint: disable=relative-import

import pycuda.autoinit
import pycuda.compiler as compiler
from pycuda.elementwise import ElementwiseKernel
import numpy as np
from artbox.tools import block_size_x, block_size_y, block, get_grid

# Definition of GPU kernels. Should start at line 25 and end at 136 (adapt docs
# otherwise!)
KERNELS = """
  #include <pycuda-complex.hpp>

  typedef pycuda::complex<float> complex_float;
  typedef pycuda::complex<double> complex_double;

__global__ void tgv_update_p(complex_float *u, complex_float *w,
                             complex_float *p, float alpha_inv, float tau_d,
                             int nx, int ny)
{
  int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
  int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

  if ((x < nx) && (y < ny)) {
    float pabs;

    complex_float u00 =              u[y*nx     + x];
    complex_float u10 = (x < nx-1) ? u[y*nx     + x+1] : u00;
    complex_float u01 = (y < ny-1) ? u[(y+1)*nx + x]   : u00;

    complex_float dxp = u10-u00;
    complex_float dyp = u01-u00;

    complex_float px = p[y*nx+x]
                     + tau_d*(dxp - w[y*nx+x]);
    complex_float py = p[(ny+y)*nx+x]
                     + tau_d*(dyp - w[(ny+y)*nx+x]);

    pabs = sqrtf(abs(px)*abs(px) + abs(py)*abs(py))*alpha_inv;
    pabs = (pabs > 1.0f) ? 1.0f/pabs : 1.0f;

    p[y*nx     +x] = px*pabs;
    p[(ny+y)*nx+x] = py*pabs;
  }
}

__global__ void tgv_update_q(complex_float *w, complex_float *q,
                             float alpha_inv, float tau_d, int nx, int ny)
{
  int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
  int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

  if ((x < nx) && (y < ny)) {
    float qabs;

    complex_float wx00 =              w[(y  )*nx+x];
    complex_float wx10 = (x < nx-1) ? w[(y  )*nx+x+1] : wx00;
    complex_float wx01 = (y < ny-1) ? w[(y+1)*nx+x]   : wx00;

    complex_float wy00 =              w[(ny+y  )*nx+x];
    complex_float wy10 = (x < nx-1) ? w[(ny+y  )*nx+x+1] : wy00;
    complex_float wy01 = (y < ny-1) ? w[(ny+y+1)*nx+x]   : wy00;

    complex_float wxx = wx10-wx00;
    complex_float wyy = wy01-wy00;
    complex_float wxy = 0.5f*(wx01-wx00+wy10-wy00);

    complex_float qxx = q[(     y)*nx+x] + tau_d*wxx;
    complex_float qyy = q[(  ny+y)*nx+x] + tau_d*wyy;
    complex_float qxy = q[(2*ny+y)*nx+x] + tau_d*wxy;

    qabs = sqrtf(abs(qxx)*abs(qxx) + abs(qyy)*abs(qyy)
           + 2.0f*abs(qxy)*abs(qxy))*alpha_inv;
    qabs = (qabs > 1.0f) ? 1.0f/qabs : 1.0f;

    q[(     y)*nx+x] = qxx*qabs;
    q[(  ny+y)*nx+x] = qyy*qabs;
    q[(2*ny+y)*nx+x] = qxy*qabs;
  }
}

__global__ void tgv_update_w(complex_float *w, complex_float *p,
                             complex_float *q, float tau_p, int nx, int ny)
{
  complex_float div1, div2;

  int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
  int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

  if ((x < nx) && (y < ny)) {
    div1 = p[y*nx+x];
    if (x < nx-1) div1 += q[(     y  )*nx+x];
    if (x > 0)    div1 -= q[(     y  )*nx+x-1];
    if (y < ny-1) div1 += q[(2*ny+y  )*nx+x];
    if (y > 0)    div1 -= q[(2*ny+y-1)*nx+x];

    div2 = p[(ny+y)*nx+x];
    if (x < nx-1) div2 += q[(2*ny+y  )*nx+x];
    if (x > 0)    div2 -= q[(2*ny+y  )*nx+x-1];
    if (y < ny-1) div2 += q[(  ny+y  )*nx+x];
    if (y > 0)    div2 -= q[(  ny+y-1)*nx+x];

    w[(   y)*nx+x] += tau_p*div1;
    w[(ny+y)*nx+x] += tau_p*div2;
  }
}

__global__ void tgv_update_u(complex_float *u, complex_float *p,
                             complex_float *Kadjv, float tau_p, int nx, int ny)
{
  int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
  int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

  if ((x < nx) && (y < ny)) {
    complex_float div = -Kadjv[y*nx+x];
    if (x < nx-1) div += p[(   y  )*nx+x];
    if (x > 0)    div -= p[(   y  )*nx+x-1];
    if (y < ny-1) div += p[(ny+y  )*nx+x];
    if (y > 0)    div -= p[(ny+y-1)*nx+x];

    u[y*nx+x] += tau_p*div;
  }
}
  """

KERNELS = KERNELS % {
    'BLOCK_SIZE_X': block_size_x,
    'BLOCK_SIZE_Y': block_size_y,
    }

MODULE = compiler.SourceModule(KERNELS)

tgv_update_u_ = \
    ElementwiseKernel("pycuda::complex<float> *u_, \
                      pycuda::complex<float> *u",
                      "u_[i] = 2.0f*u[i] - u_[i]",
                      "extragradient_update",
                      preamble="#include <pycuda-complex.hpp>")

tgv_update_u_2 = \
    ElementwiseKernel("pycuda::complex<float> *u_, \
                      pycuda::complex<float> *u, \
                      pycuda::complex<float> *uold",
                      "u_[i] = 2.0f*u[i] - uold[i]",
                      "extragradient_update",
                      preamble="#include <pycuda-complex.hpp>")

tgv_update_v_func = \
    ElementwiseKernel("pycuda::complex<float> *v, \
                      pycuda::complex<float> *Ku, \
                      pycuda::complex<float> *f, float w1, float w2",
                      "v[i] = w1*v[i] + w2*(Ku[i] - f[i])",
                      "tgv_update_v",
                      preamble="#include <pycuda-complex.hpp>")

tgv_update_u_func = MODULE.get_function("tgv_update_u")
tgv_update_p_func = MODULE.get_function("tgv_update_p")
tgv_update_q_func = MODULE.get_function("tgv_update_q")
tgv_update_u_func = MODULE.get_function("tgv_update_u")
tgv_update_w_func = MODULE.get_function("tgv_update_w")
tgv_update_u_ = tgv_update_u_
tgv_update_u_2 = tgv_update_u_2
tgv_update_w_2 = tgv_update_u_2
tgv_update_w_ = tgv_update_u_


# TGV wrappers
def tgv_update_p(u, w, p, tau_d, alpha):
    """Update p

    Args:
        u (gpuarray): u.
        w (gpuarray): w.
        p (gpuarray): p.
        tau_d (float): tau_d.
        alpha (float): alpha.
    """
    tgv_update_p_func(u, w, p, np.float32(1.0/alpha), np.float32(tau_d),
                      np.int32(u.shape[0]), np.int32(u.shape[1]),
                      block=block, grid=get_grid(u))


def tgv_update_q(w, q, tau_d, alpha):
    """Update q

    Args:
        w (gpuarray): w.
        q (gpuarray): q.
        tau_d (float): tau_d.
        alpha (float): alpha.
    """
    tgv_update_q_func(w, q, np.float32(1.0/alpha), np.float32(tau_d),
                      np.int32(w.shape[0]), np.int32(w.shape[1]),
                      block=block, grid=get_grid(w))


def tgv_update_v(v, Ku, f, tau_d, lin_constr=False):
    """Update v

    Args:
        v (gpuarray): v.
        Ku (gpuarray): Ku.
        f (gpuarray): f.
        tau_d (float): tau_d.
        lin_constr (bool): lin_constr.
    """
    if not lin_constr:
        tgv_update_v_func(v, Ku, f, np.float32(1.0/(1.0 + tau_d)),
                          np.float32(tau_d/(1.0 + tau_d)))
    else:
        tgv_update_v_func(v, Ku, f, np.float32(1.0), np.float32(tau_d))


def tgv_update_u(u, p, f, tau_p):
    """Update u

    Args:
        u (gpuarray): u.
        p (gpuarray): p.
        f (gpuarray): f.
        tau_p (float): tau_p.
    """
    tgv_update_u_func(u, p, f, np.float32(tau_p),
                      np.int32(u.shape[0]), np.int32(u.shape[1]),
                      block=block, grid=get_grid(u))


def tgv_update_w(w, p, q, tau_p):
    """Update w

    Args:
        w (gpuarray): w.
        p (gpuarray): p.
        q (gpuarray): q.
        tau_p (float): tau_p.
    """
    tgv_update_w_func(w, p, q, np.float32(tau_p),
                      np.int32(w.shape[0]), np.int32(w.shape[1]),
                      block=block, grid=get_grid(w))
