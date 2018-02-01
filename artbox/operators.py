# Copyright 2018 Stefan Kroboth
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
"""Encoding matrix Operator.

`Operator` object which provides a forward operator (`apply`) and an adjoint
operator (`adjoint`) which implement operations with the encoding matrix.

Author: Stefan Kroboth <stefan.kroboth@gmail.com>
"""

# pylint: disable=relative-import
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals

from __future__ import print_function
import pycuda.gpuarray as gpuarray
import pycuda.compiler as compiler
import pycuda.curandom as curandom
import numpy as np
from artbox.tools import next_power_of_2, gpuarray_copy, dotc_gpu

KERNELS = """
  #include <pycuda-complex.hpp>

  typedef pycuda::complex<%(PRECISION)s> complex;
  //typedef pycuda::complex<float> complex_float;
  //typedef pycuda::complex<double> complex_double;

  %(G_MAT_PROVIDED)s
  %(B0_MAT_PROVIDED)s

  __device__ complex sinc(%(PRECISION)s x)
  {
    complex out;
    //if(abs(x) <= 1e-8)
    if(x == 0)
    {
      out = 1.0;
    }
    else
    {
      out = sin(x)/x;
    }
    return out;
  }

  __device__ complex phi_elem(unsigned iK, unsigned iP, %(PRECISION)s* k_mat,
                              %(PRECISION)s* psi_mat, %(PRECISION)s* g_mat,
                              %(PRECISION)s* b0, %(PRECISION)s* ktime)
  {
    %(PRECISION)s temp = 0;
  #ifdef G_MAT
    %(PRECISION)s row1 = 0;
    %(PRECISION)s row2 = 0;
    %(PRECISION)s row3 = 0;
  #endif
    complex result(0.0,0.0);
    if(iP < %(P_RES)s && iK < %(K_RES)s)
    {
      for(int i = 0; i < %(NUM_SEM)s; i++)
      {
        temp += k_mat[iK + i * %(K_RES)s] * psi_mat[iP + i * %(P_RES)s];
  #ifdef G_MAT
        row1 += k_mat[iK + i * %(K_RES)s]
                * g_mat[iP + i * %(P_RES)s];
        row2 += k_mat[iK + i * %(K_RES)s]
                * g_mat[iP + i * %(P_RES)s +     %(P_RES)s * %(NUM_SEM)s];
        row3 += k_mat[iK + i * %(K_RES)s]
                * g_mat[iP + i * %(P_RES)s + 2 * %(P_RES)s * %(NUM_SEM)s];
  #endif
      }
  #ifdef B0_MAT
      %(PRECISION)s b0_c = b0[iP] * ktime[iK];
  #endif
  #ifdef G_MAT
      result = complex(cos(temp), sin(-temp)) * sinc(%(W1)s * row1)
               * sinc(%(W2)s * row2) * sinc(%(W3)s * row3)
  #else
      result = complex(cos(temp), sin(-temp))
  #endif
  #ifdef B0_MAT
      / complex(cos(b0_c), sin(-b0_c));
  #else
      ;
  #endif

      result = complex(cos(temp), sin(-temp));
    }
    return result;
  }

  __global__ void reshape_cmat(complex* CmatIn, complex* CmatOut)
  {
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
    complex temp(0);

    if (x < %(P_RES)s)
      temp = CmatIn[y * %(P_RES)s + x];

    CmatOut[y*%(P_RES32)s + x] = temp;
  }

  __global__ void multiply_E_m(complex* mc, complex* erg, %(PRECISION)s* k_mat,
                               %(PRECISION)s* psi_mat, %(PRECISION)s* g_mat,
                               %(PRECISION)s* b0, %(PRECISION)s* ktime,
                               int fst_row)
  {
    // hold the partial sums
    __shared__ complex partial_row_sum[%(MAX_NUM_THREADS_PER_BLOCK)s];
    __syncthreads();
    //calculate the product of the current row with the vector x with a stride
    complex phi_prefetch[%(PHI_PREFETCH_ELEM)s];
    //equal to the amount of threads
    for(unsigned jump = blockIdx.y + fst_row;
        jump < %(K_RES)s;
        jump += %(HOP_E)s)
    {
      complex sum(0.0, 0.0);
      unsigned elem = 0;
      for (unsigned index = threadIdx.x;
           index < %(P_RES)s;
           index += blockDim.x)
      {
        phi_prefetch[elem] = phi_elem(jump, index, k_mat, psi_mat, g_mat, b0,
                                      ktime);
        elem++;
      }

      for(unsigned coil = 0; coil < %(N_COILS)s; coil++)
      {
        sum = complex(0.0,0.0);
        elem = 0;
        for (unsigned index = threadIdx.x;
             index < %(P_RES)s;
             index += blockDim.x)
        {
          sum += phi_prefetch[elem] * mc[index + %(P_RES32)s*coil];
          elem++;
        }

        partial_row_sum[threadIdx.x] = sum;

        // reduce the partial sums
        for (unsigned counter = blockDim.x >> 1; counter != 0; counter >>= 1)
        {
          __syncthreads();
          if (threadIdx.x < counter)
            partial_row_sum[threadIdx.x] +=
              partial_row_sum[threadIdx.x + counter];
        }

        // write the result into the output Vector
        if (threadIdx.x == 0)
          erg[jump + %(K_RES)s*coil] = partial_row_sum[0];
      }
    }
  }

  __global__ void multiply_cmat_m(complex* m, complex* mc, complex* c_mat)
  {
    unsigned long np_index = threadIdx.x + blockDim.x*blockIdx.x;
    if(np_index >= %(P_RES)s)
      return;

    for(unsigned i = 0; i < %(N_COILS)s; i++)
    {
      mc[i*%(P_RES32)s + np_index] = c_mat[np_index + i*%(P_RES32)s]
                                     * m[np_index];
    }
  }

  __global__ void multiply_EH_m(complex* m, complex* erg, complex* c_mat,
                                %(PRECISION)s* k_mat, %(PRECISION)s* psi_mat,
                                %(PRECISION)s* g_mat, %(PRECISION)s* b0,
                                %(PRECISION)s* ktime, int fst_row)
  {
    __shared__ complex partial_row_sum[%(MAX_NUM_THREADS_PER_BLOCK)s];
    __shared__ complex cmat_cache[%(N_COILS)s];
    __syncthreads();
    complex phi_cache[512];
    // number of elements that are to be cached
    unsigned phi_cache_length = %(K_RES)s / blockDim.x ;

    for(unsigned jump = blockIdx.y + fst_row;
        jump < %(P_RES)s;
        jump += %(HOP_EH)s)
    {
      // current position in the cmat matrix
      //unsigned tmp = threadIdx.x*%(P_RES32)s + jump;

      // prefetch all cmat values needed for this row into the shared memory
      if(threadIdx.x < %(N_COILS)s)
        cmat_cache[threadIdx.x] = c_mat[threadIdx.x*%(P_RES32)s+jump];
        //cmat_cache[threadIdx.x] = c_mat[tmp];

      // threads need to be synced at this point to make sure that all
      // cmat values are in the shared
      // memory before the computation starts
      __syncthreads();

      // prefetch all needed values of phi.
      // Since some values are needed more then once, this reduces the
      // number of accesses to the memory.
      for(unsigned i = 0; i < phi_cache_length; i++)
        phi_cache[i] = phi_elem(i*blockDim.x+threadIdx.x, jump, k_mat, psi_mat,
                                g_mat, b0, ktime);

      complex sum(0.0,0.0);

      // compute the product of the element of E^H and the input vector and sum
      // up the results
      for (unsigned j = 0; j < phi_cache_length; j++)
        for (unsigned k = 0; k < %(N_COILS)s; k++)
          sum += conj(cmat_cache[k] * phi_cache[j])
            * m[k*%(K_RES)s + j*blockDim.x + threadIdx.x];

      partial_row_sum[threadIdx.x] = sum;

      // reduce the partial sums
      for (unsigned counter = blockDim.x >> 1; counter != 0; counter >>= 1)
      {
        __syncthreads();
        if (threadIdx.x < counter)
          partial_row_sum[threadIdx.x] +=
            partial_row_sum[threadIdx.x + counter];
      }

      // write the result into the output vector
      if(threadIdx.x == 0)
        erg[jump] = partial_row_sum[0];

      __syncthreads();
    }
  }
  """


class Operator(object):
    """Create an operator object for the encoding matrix :math:`E`. This object
    is used in reconstruction algorithms and provides a forward operator
    (`apply`, which implements :math:`Em`) and an adjoint operator (`adjoint`,
    which implements :math:`E^Hm`). The operator can only be applied to
    vectors.

    Args:
        data (ReconData): Loaded dataset.
        double (bool): Indicates whether computations should be performed
            with double precision or not.
        max_threads (int): Number of GPU threads.
        norm_div (float): Divide norm of operator by this value (may speed
            up convergence).
        divide (int): Divide problem into `divide` subproblems (can be
            beneficial if the kernels get terminated because the take too
            long to run). This applies to both forward and adjoint
            operations.
        hop (int): Decrease this parameter if your code crashes because
            the given problem is too big. This applies to both forward and
            adjoint operations.
        divide_adjoint (int): Same as `divide`, but only for adjoint
            operator.
        divide_forward (int): Same as `divide` but only for forward
            operator.
        hop_adjoint (int): Same as `hop` but only for adjoint operator.
        hop_forward (int): Same as `hop` but only for forward operator.
        show_kernel_params (int): Print out kernel parameters while
            initializing the object.
        verbose (int): Be verbose.
    """
    def __init__(self, data, double=False, max_threads=256, norm_div=1,
                 divide=1, hop=8192, divide_adjoint=None, divide_forward=None,
                 hop_adjoint=None, hop_forward=None, show_kernel_params=False,
                 verbose=False):
        """Constructor
        """
        self.op_type = "ENCODINGMAT"
        self.data = data
        self._norm_div = norm_div
        self._verbose = verbose

        if double:
            self.precision_str = 'double'
            self.precision_complex = np.complex128
            self.precision_real = np.float64
        else:
            self.precision_str = 'float'
            self.precision_complex = np.complex64
            self.precision_real = np.float32

        #: Kernel parameters
        self.kp = dict()
        self.kp['max_threads'] = max_threads
        if divide_forward is not None:
            #  self.divide_E = int(divide_forward)
            self.kp['divide_E'] = int(divide_forward)
        else:
            self.kp['divide_E'] = int(divide)

        if divide_adjoint is not None:
            self.kp['divide_EH'] = int(divide_adjoint)
        else:
            self.kp['divide_EH'] = int(divide)

        if hop_adjoint is not None:
            self.kp['hop_E'] = int(hop_forward)
        else:
            self.kp['hop_E'] = int(hop)

        if hop_adjoint is not None:
            self.kp['hop_EH'] = int(hop_adjoint)
        else:
            self.kp['hop_EH'] = int(hop)

        self.kp['block_E'] = int(float(self.kp['hop_E']) /
                                 self.kp['divide_E'])
        self.kp['block_EH'] = int(float(self.kp['hop_EH']) /
                                  self.kp['divide_EH'])
        self.kp['threads_E'] = int(max_threads)
        self.kp['grid_E'] = int(self.kp['block_E'])
        self.kp['grid_EH'] = int(self.kp['block_EH'])
        self.kp['grid_MC'] = int(np.ceil(float(self.data.p_res) /
                                         max_threads))
        self.kp['threads_MC'] = int(max_threads)

        if show_kernel_params:
            self.print_kernel_params()

        # temporary variable holding the input vector m multplied by each coil
        # sensitivity

        #: Data on the GPU
        self.dgpu = dict()
        self.dgpu['mc'] = gpuarray.zeros((data.p_res32, data.nC),
                                         self.precision_complex)
        self.dgpu['k'] = gpuarray.to_gpu(data.k)
        self.dgpu['c_mat'] = gpuarray.to_gpu(data.c_mat)
        self.dgpu['psi'] = gpuarray.to_gpu(data.psi)
        if data.recondata is not None:
            self.dgpu['recondata'] = gpuarray.to_gpu(data.recondata)

        if data.b0 is None:
            self._b0_provided = ''
            self.dgpu['b0_mat'] = gpuarray.to_gpu(np.zeros(1))
            self.dgpu['ktime_mat'] = gpuarray.to_gpu(np.zeros(1))
        else:
            self._b0_provided = '#define B0_MAT'
            self.dgpu['b0_mat'] = gpuarray.to_gpu(data.b0.flatten())
            self.dgpu['ktime_mat'] = gpuarray.to_gpu(data.ktime.flatten())

        if data.G_mat is None:
            self._g_mat_provided = ''
            self.dgpu['g_mat'] = gpuarray.to_gpu(np.zeros(1))
        else:
            self._g_mat_provided = '#define G_MAT'
            self.dgpu['g_mat'] = gpuarray.to_gpu(data.G_mat.flatten())

        self._multiply_E_m_func = None
        self._multiply_EH_m_func = None
        self._reshape_cmat_func = None
        self._multiply_cmat_m_func = None

        self._compile_recon_kernels()

    def _compile_recon_kernels(self):
        """Compile reconstruction kernels

        Replaces placeholders in `KERNEL` with parameters derived from the data
        and user input and compiles them. This way it is possible to adapt the
        code to the data before compilation. This provides more information for
        the compiler and hence potentially leads to faster code.
        """
        loc_kernels = KERNELS % {
            'P_RES':                     int(self.data.p_res),
            'K_RES':                     int(self.data.nT),
            'PRECISION':                 self.precision_str,
            'W1':                        float(self.data.w[0]/2),  # check /2
            'W2':                        float(self.data.w[1]/2),
            'W3':                        float(self.data.w[2]/2),
            'MAX_NUM_THREADS_PER_BLOCK': int(self.kp['max_threads']),
            'PHI_PREFETCH_ELEM':
                int(next_power_of_2(self.data.p_res / self.kp['threads_E'])),
            'HOP_E':                     int(self.kp['hop_E']),
            'N_COILS':                   int(self.data.nC),
            'P_RES32':                   int(self.data.p_res32),
            'GRID_E':                    int(self.kp['grid_E']),
            'THREADS_E':                 int(self.kp['threads_E']),
            'BLOCK_E':                   int(self.kp['block_E']),
            'GRID_MC':                   int(self.kp['grid_MC']),
            'THREADS_MC':                int(self.kp['threads_MC']),
            'HOP_EH':                    int(self.kp['hop_EH']),
            'NUM_SEM':                   int(self.data.nF),
            'B0_MAT_PROVIDED':           self._b0_provided,
            'G_MAT_PROVIDED':            self._g_mat_provided,
        }

        # Compile the kernels
        module = compiler.SourceModule(loc_kernels)

        # Assign the references to the kernels to class members
        self._multiply_E_m_func = module.get_function("multiply_E_m")
        self._multiply_EH_m_func = module.get_function("multiply_EH_m")
        self._reshape_cmat_func = module.get_function("reshape_cmat")
        self._multiply_cmat_m_func = module.get_function("multiply_cmat_m")

    def print_kernel_params(self):
        """Print kernel parameters to the terminal

        This may be helpful if stdout is piped to a file for documentation
        or debugging purposes.
        """
        print("Kernel parameters:")
        print("  block_E:    " + str(self.kp['block_E']))
        print("  block_EH:   " + str(self.kp['block_EH']))
        print("  threads_E:  " + str(self.kp['threads_E']))
        print("  grid_E:     " + str(self.kp['grid_E']))
        print("  grid_EH:    " + str(self.kp['grid_EH']))
        print("  grid_MC:    " + str(self.kp['grid_MC']))
        print("  threads_MC: " + str(self.kp['threads_MC']))

    def apply(self, m, result):
        """Multiply the encoding matrix :math:`E` with the input vector
        :math:`m`. `result` will be modified in-place.

        :math:`result = Em`

        Args:
            m (gpuarray): input vector :math:`m`
            result (gpuarray): result of :math:`Em`
        """

        #: multiply m with each coil sensitivity
        self._multiply_cmat_m_func(m,
                                   self.dgpu['mc'],
                                   self.dgpu['c_mat'],
                                   block=(self.kp['max_threads'], 1, 1),
                                   grid=(self.kp['grid_MC'], 1))
        #: solve subproblems
        for i in range(0, self.kp['divide_E']):
            self._multiply_E_m_func(self.dgpu['mc'],
                                    result,
                                    self.dgpu['k'],
                                    self.dgpu['psi'],
                                    self.dgpu['g_mat'],
                                    self.dgpu['b0_mat'],
                                    self.dgpu['ktime_mat'],
                                    np.int32(i * self.kp['block_E']),
                                    block=(self.kp['max_threads'], 1, 1),
                                    grid=(1, self.kp['grid_E']))

    def adjoint(self, m, result):
        """Multiply the adjoint of the encoding matrix (:math:`E^H`) with the
        input vector :math:`m`. `result` will be modified in-place.

        :math:`result = E^Hm`

        Args:
            m (gpuarray): input vector :math:`m`
            result (gpuarray): result of :math:`E^Hm`
        """
        #: solve the subproblems
        for i in range(0, self.kp['divide_EH']):
            self._multiply_EH_m_func(m,
                                     result,
                                     self.dgpu['c_mat'],
                                     self.dgpu['k'],
                                     self.dgpu['psi'],
                                     self.dgpu['g_mat'],
                                     self.dgpu['b0_mat'],
                                     self.dgpu['ktime_mat'],
                                     np.int32(i * self.kp['block_EH']),
                                     block=(self.kp['max_threads'], 1, 1),
                                     grid=(1, self.kp['grid_EH']))
            # cuda.Context.synchronize()

    def norm_est(self, u, iters=10):
        """Estimates norm of the operator with a power iteration.

        Args:
            u (gpuarray): input array
            iters (int): number of iterations
        """
        if self._verbose:
            print("Estimating Norm...")

        u_temp = gpuarray_copy(u)
        result = gpuarray.zeros([self.data.nC, self.data.nT],
                                self.precision_complex, order='F')

        for _ in range(0, iters):
            dot_tmp = dotc_gpu(u_temp)
            u_temp /= np.sqrt(np.abs(dot_tmp))
            self.apply(u_temp, result)
            self.adjoint(result, u_temp)
            normsqr = dotc_gpu(u_temp)
        return np.sqrt(np.abs(normsqr)/self._norm_div)

    def test_adjoint(self, iters=5):
        """Test the adjoint operator.

        Args:
            iters (int): number of iterations
        """
        src_shape = (self.data.nX1, self.data.nX2, 1)
        dest_shape = (self.data.nT, self.data.nC)
        u = gpuarray.zeros(src_shape, self.precision_complex, order='F')
        ut = gpuarray.zeros(src_shape, self.precision_real, order='F')
        Ku = gpuarray.zeros(dest_shape, self.precision_complex, order='F')
        v = gpuarray.zeros(dest_shape, self.precision_complex, order='F')
        vt = gpuarray.zeros(dest_shape, self.precision_real, order='F')
        Kadv = gpuarray.zeros(src_shape, self.precision_complex, order='F')

        generator = curandom.XORWOWRandomNumberGenerator()
        errors = []

        try:
            i = 0
            for i in range(iters):
                # randomness
                generator.fill_uniform(ut)
                generator.fill_uniform(vt)
                v = gpuarray_copy(vt.astype(self.precision_complex))
                u = gpuarray_copy(ut.astype(self.precision_complex))

                # apply operators
                self.apply(u, Ku)
                self.adjoint(v, Kadv)

                scp1 = dotc_gpu(Ku, v)
                scp2 = dotc_gpu(u, Kadv)
                n_Ku = dotc_gpu(Ku)
                n_Kadv = dotc_gpu(Kadv)
                n_u = dotc_gpu(u)
                n_v = dotc_gpu(v)

                errors.append(np.abs(scp1-scp2))

            print("Test " + str(i) + ": <Ku,v>=" + str(scp1) + ", <u,Kadv>=" +
                  str(scp2) + ", Error=" + str(np.abs(scp1-scp2)) +
                  ", Relative Error=" +
                  str((scp1-scp2)/(n_Ku*n_v + n_Kadv*n_u)))
        except KeyboardInterrupt:
            if len(errors) == 0:
                errors = -1
        finally:
            print("Mean Error: " + repr(np.mean(errors)))
            print("Standarddeviation: " + repr(np.std(errors)))
        return i
