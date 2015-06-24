# -*- coding: utf-8 -*-

"""
Python functions to cuBLAS
For documentation see:
http://docs.nvidia.com/cuda/cublas/index.htm
"""

import ctypes
import numpy
import pycublas
import pycuda.gpuarray

class _ndarray_ptr(object):
    def __init__(self, ndarray):
        self.data = ndarray #Keep the array alive
        self.ptr = ndarray.ctypes.data
    def get(self):
        return self.data
    @property
    def dtype(self):
        return self.data.dtype

def _isScalar(s):
    return isinstance(s, (int, float, complex))

def _isArray(a):
    return isinstance(a, (pycuda.gpuarray.GPUArray, numpy.ndarray) )

def _isOnGPU(array):
    return isinstance(array, pycuda.gpuarray.GPUArray)

_valid_GPU_types = ['float32','float64','complex64','complex128']

def _toGPU(data, new_dtype):
    if numpy.dtype(new_dtype).name not in _valid_GPU_types:
        new_dtype = 'float64'

    if _isScalar(data):
        return pycuda.gpuarray.to_gpu( numpy.array([data], dtype=new_dtype) )

    elif _isOnGPU(data):
        if data.flags.f_contiguous:
            if data.dtype == new_dtype:
                return data
            else:
                return data.astype(new_dtype)
        else:
            return _toGPU(numpy.asfortranarray(data.get(), new_dtype))

    elif isinstance(data, numpy.ndarray):
        return pycuda.gpuarray.to_gpu( numpy.asfortranarray(data, new_dtype) )
    else:
        raise TypeError("data must be array or scalar")


class pycublasContext(object):
    def __init__(self):
        self._handle = pycublas.cublasHandle_t()
        self._cublasStatus = pycublas.cublasCreate(self._handle)
        
        self.CheckStatusFunction = None
        
        self._returnToHost = True
        self._autoCast = True
        
    def __del__(self):
        self.cublasStatus = pycublas.cublasDestroy(self._handle)

    @property
    def autoCast(self):
        '''
        True  : Automatic cast of arrays and scalars to the common dtype
        False : Do not check array dtypes,
                scalars are always casted to the appropriate type
        '''
        return self._autoCast
    @autoCast.setter
    def autoCast(self, value):
        self._autoCast = bool(value)
            
    def _AutoCaster(self, *args):
        '''
        args = self._AutoCaster(*args)
        return scalars as numpy.darrays and
               arrays as pycuda.gpuarray.GPUArray
        '''
        if all( [_isScalar(x) for x in args] ):
            #all are scalars so we cast them to the same type
            _areComplex = any( [isinstance(x,complex) for x in args] )
            new_dtype = 'complex128' if _areComplex else 'float64'
            return [numpy.array([x], dtype=new_dtype) for x in args]
        elif not self.autoCast:
            #cast scalars to the fist array found
            new_dtype  = next( (x.dtype for x in args if _isArray(x)) )
            return [numpy.array([x], dtype=new_dtype) if _isScalar(x)
                    else _toGPU(x, x.dtype) 
                    for x in args]
        else: #autoCast
            _areComplex = any( [isinstance(x,complex) if _isScalar(x) 
                                else ('complex' in x.dtype.name)
                                for x in args] )
            _areSingle = all( [True if _isScalar(x) 
                              else x.dtype.name in ['float32','complex64']
                              for x in args] )
            new_dtype = {(False,True ):'float32',
                         (False,False):'float64',
                         (True, True ):'complex64',
                         (True, False):'complex128'}[(_areComplex, _areSingle)]
            return [numpy.array([x], dtype=new_dtype) if _isScalar(x) 
                    else _toGPU(x, new_dtype) 
                    for x in args]

    def _getOPs(self, *args, **kargs):
        op_dict = {'N': pycublas.cublasOperation_t.CUBLAS_OP_N,
                   'T': pycublas.cublasOperation_t.CUBLAS_OP_T,
                   'C': pycublas.cublasOperation_t.CUBLAS_OP_C}
        valid_keys = kargs['valid'] if 'valid' in kargs else op_dict.keys()
        if all([op in valid_keys for op in args]):
            return (op_dict[op] for op in args)
        else:
            valid_string = ', '.join("'%s'"% s for s in valid_keys[:-1]) + \
                           " or '%s'" % valid_keys[-1]
            raise ValueError("op must be %s" % valid_string)

    def _getSIDESs(self, *args):
        side_dict = {'L': pycublas.cublasSideMode_t.CUBLAS_SIDE_LEFT,
                     'R': pycublas.cublasSideMode_t.CUBLAS_SIDE_RIGHT}
        if all([side in side_dict.keys() for side in args]):
            return (side_dict[side] for side in args)
        else:
            raise ValueError("side must be 'L' (left) or 'R' (rigth)")
            
    def _getFILL_MODEs(self, *args):
        fm_dict = {'U': pycublas.cublasFillMode_t.CUBLAS_FILL_MODE_UPPER,
                   'L': pycublas.cublasFillMode_t.CUBLAS_FILL_MODE_LOWER}
        if all([fm in fm_dict.keys() for fm in args]):
            return (fm_dict[fm] for fm in args)
        else:
            raise ValueError("fillMode must be 'U' (upper) or 'L' (lower)")

    @property
    def returnToHost(self):
        '''
        if True return arrays are instances of numpy.ndarray
        returnToHost = not(returnToDevice)
        '''
        return self._returnToHost
    @returnToHost.setter
    def returnToHost(self, value):
        self._returnToHost = bool(value)
    @property
    def returnToDevice(self):
        '''
        if True return arrays are instances of pycuda.gpuarray.GPUArray
        returnToDevice = not(returnToHost)
        '''
        return not self._returnToHost
    @returnToDevice.setter
    def returnToDevice(self, value):
        self._returnToHost = not bool(value)

    def _return(self, data):
        if self.returnToDevice:
            return data
        elif self.returnToHost:
            return data.get() #TODO use cublasGetVector / Matrix
    
    ## cublasStatus Check ##
    @property
    def cublasStatus(self):
        return self._cublasStatus
    @cublasStatus.setter
    def cublasStatus(self, status):
        if isinstance(status, pycublas.cublasStatus_t):
            self._cublasStatus = status
        if callable(self.CheckStatusFunction):
            self.CheckStatusFunction(self._cublasStatus)
        
    ## cuBLAS Helper Functions ##
    @property
    def Version(self):
        version = ctypes.c_int()
        self.cublasStatus = pycublas.cublasGetVersion(self._handle, version)
        return version.value

    # cublasPointerMode
    @property
    def pointerMode(self):
        pMode = pycublas.c_cublasPointerMode_t()
        self.cublasStatus = pycublas.cublasGetPointerMode(self._handle, pMode)
        return pycublas.cublasPointerMode_t(pMode.value)
    @pointerMode.setter
    def pointerMode(self, mode):
        if isinstance(mode, pycublas.cublasPointerMode_t):
            mode = mode.value
        if mode in ['CUBLAS_POINTER_MODE_HOST', 0, 'Host', 'HOST']:
            mode = 0
        elif mode in ['CUBLAS_POINTER_MODE_DEVICE', 1, 'Device', 'DEVICE']:
            mode = 1
        else:
            mode = self.pointerMode.value
        self.cublasStatus = pycublas.cublasSetPointerMode(self._handle, mode)

    # cublasAtomicsMode       
    @property
    def atomicsMode(self):
        aMode = pycublas.c_cublasAtomicsMode_t()
        self.cublasStatus = pycublas.cublasGetAtomicsMode(self._handle, aMode)
        return pycublas.cublasAtomicsMode_t(aMode.value)
    @atomicsMode.setter
    def atomicsMode(self, mode):
        if isinstance(mode, pycublas.cublasAtomicsMode_t):
            mode = mode.value
        if mode in ['CUBLAS_ATOMICS_NOT_ALLOWED', 0, False, 'NOT_ALLOWED']:
            mode = 0
        elif mode in ['CUBLAS_ATOMICS_ALLOWED', 1, True, 'ALLOWED']:
            mode = 1
        else:
            mode = self.atomicsMode.value
        self.cublasStatus = pycublas.cublasSetAtomicsMode(self._handle, mode)

    ## cuBLAS Level-1 Functions ##
    
    # cublasI_amax
    def I_amax(self, X, incx = 1):
        X = _toGPU(X, X.dtype)
    
        I_amax_function = {'float32'    : pycublas.cublasIsamax,
                           'float64'    : pycublas.cublasIdamax,
                           'complex64'  : pycublas.cublasIcamax,
                           'complex128' : pycublas.cublasIzamax
                           }[X.dtype.name]
        result = ctypes.c_int()
        
        self.cublasStatus = I_amax_function(self._handle, X.size,
                                            X.ptr, incx, result)
        return result.value - 1        

    # cublasI_amin        
    def I_amin(self, X, incx = 1):
        X = _toGPU(X, array.dtype)
      
        I_amin_function = {'float32'    : pycublas.cublasIsamin,
                           'float64'    : pycublas.cublasIdamin,
                           'complex64'  : pycublas.cublasIcamin,
                           'complex128' : pycublas.cublasIzamin
                           }[X.dtype.name]
        result = ctypes.c_int()
        
        self.cublasStatus = I_amin_function(self._handle, X.size,
                                            X.ptr, incx, result)
        return result.value - 1  

    # cublas_asum         
    def asum(self, X, incx = 1):
        X = _toGPU(X, X.dtype)
                  
        asum_function = {'float32'    : pycublas.cublasSasum, 
                         'float64'    : pycublas.cublasDasum,
                         'complex64'  : pycublas.cublasScasum,
                         'complex128' : pycublas.cublasDzasum
                         }[X.dtype.name]
        result_type = {'float32'    : ctypes.c_float,
                       'float64'    : ctypes.c_double,
                       'complex64'  : ctypes.c_float,
                       'complex128' : ctypes.c_double
                       }[X.dtype.name]   
                         
        result = result_type()
        self.cublasStatus = asum_function(self._handle, X.size,
                                          X.ptr, incx, result)
        return result.value

    # cublas_axpy         
    def axpy(self, alpha, X, Y, incx = 1, incy = 1):
        '''
        Y = alpha * X + Y
        '''
        Y, alpha, X = self._AutoCaster(Y, alpha, X)

        if _isOnGPU(alpha):
            self.pointerMode = 'DEVICE'
        else:
            self.pointerMode = 'HOST'
            alpha = _ndarray_ptr(alpha)
          
        axpy_function = {'float32'    : pycublas.cublasSaxpy, 
                         'float64'    : pycublas.cublasDaxpy,
                         'complex64'  : pycublas.cublasCaxpy,
                         'complex128' : pycublas.cublasZaxpy
                         }[Y.dtype.name]
        self.cublasStatus = axpy_function(self._handle, Y.size,
                                          alpha.ptr,
                                          X.ptr, incx,
                                          Y.ptr, incy)
        return self._return(Y)
    
    #TODO cublas_copy

    # cublas_dot         
    def dot(self, X, Y, incx = 1, incy = 1, cc = False):
        '''
        X.Y
        if cc (complex conjugate) = True
        X.Y*
        '''
        Y, X = self._AutoCaster(Y, X)
        if 'float' in Y.dtype.name:  
            dot_function = {'float32' : pycublas.cublasSdot, 
                            'float64' : pycublas.cublasDdot
                           }[Y.dtype.name]
        else: # complex
            dot_function = {('complex64' , False) : pycublas.cublasCdotu,
                            ('complex128', False) : pycublas.cublasZdotu,
                            ('complex64' , True)  : pycublas.cublasCdotc,
                            ('complex128', True)  : pycublas.cublasZdotc,
                           }[(Y.dtype.name, cc)]

        result = _ndarray_ptr( numpy.array([0], dtype=Y.dtype) )
        self.cublasStatus = dot_function(self._handle, Y.size,
                                         X.ptr, incx,
                                         Y.ptr, incy,
                                         result.ptr)
        return result.data[0]
            
    # cublas_nrm2         
    def nrm2(self, X, incx = 1):
        """
        Eucledian norm
        """
        X = _toGPU(X, X.dtype)
                  
        nrm2_function = {'float32'    : pycublas.cublasSnrm2, 
                         'float64'    : pycublas.cublasDnrm2,
                         'complex64'  : pycublas.cublasScnrm2,
                         'complex128' : pycublas.cublasDznrm2
                         }[X.dtype.name]
        result_type = {'float32'    : ctypes.c_float,
                       'float64'    : ctypes.c_double,
                       'complex64'  : ctypes.c_float,
                       'complex128' : ctypes.c_double
                       }[X.dtype.name]   
                         
        result = result_type()
        self.cublasStatus = nrm2_function(self._handle, X.size,
                                          X.ptr, incx, result)
        return result.value

     # cublas_rot
    def rot(self, X, Y, c, s, incx = 1, incy = 1):
        '''
        (X, Y) = rot(X, Y, c, s, incx = 1, incy = 1)
        This function applies Givens rotation matrix

        G = [[  c, s],
             [-s*, c]]

        to vectors X and Y.
        Hence, the result is X[k] =   c * X[k] + s * Y[j]
                         and Y[j] = - s * X[k] + c * Y[j]
        where k = i * incx  and j = i * incy

        if c is complex, only the real part is used
        '''
        Y, X, c, s = self._AutoCaster(Y, X, c, s)
        if 'float' in Y.dtype.name:
            dot_function = {'float32' : pycublas.cublasSrot,
                            'float64' : pycublas.cublasDrot
                           }[Y.dtype.name]
        else: # complex
            s_complex = (s[0].imag != 0)
            dot_function = {('complex64' , True) : pycublas.cublasCrot,
                            ('complex128', True) : pycublas.cublasZrot,
                            ('complex64' , False): pycublas.cublasCsrot,
                            ('complex128', False): pycublas.cublasZdrot,
                           }[(Y.dtype.name, s_complex)]

        s = _ndarray_ptr(s)
        c = _ndarray_ptr(c.real)
        self.cublasStatus = dot_function(self._handle, Y.size,
                                         X.ptr, incx,
                                         Y.ptr, incy,
                                         c.ptr, s.ptr)
        return self._return(X), self._return(Y)

     # cublas_rotg
    def rotg(self, a, b):
        '''
        (c, s) = rotg(self, a, b)
        This function constructs the Givens rotation matrix

        G = [[  c, s],
             [-s*, c]]

        such that G.[a,b] = [r,0]
        '''
        a, b = self._AutoCaster(a, b)

        if _isOnGPU(a) or _isOnGPU(b):
            self.pointerMode = 'DEVICE'
            a = _toGPU(a, a.dtype)
            b = _toGPU(b, a.dtype)
            s = _toGPU(0, a.dtype)
            c = _toGPU(0, a.real.dtype)
        else:
            self.pointerMode = 'HOST'
            a = _ndarray_ptr(a)
            b = _ndarray_ptr(b)
            s = _ndarray_ptr( numpy.array([0], dtype=a.dtype) )
            c = _ndarray_ptr( numpy.array([0], dtype=a.data.real.dtype) )
        rotg_function = {'float32'    : pycublas.cublasSrotg,
                         'float64'    : pycublas.cublasDrotg,
                         'complex64'  : pycublas.cublasCrotg,
                         'complex128' : pycublas.cublasZrotg
                         }[a.dtype.name]

        self.cublasStatus = rotg_function(self._handle,
                                          a.ptr, b.ptr,
                                          c.ptr, s.ptr)
        if self.returnToDevice:
            return _toGPU(c, c.dtype), _toGPU(s, s.dtype)
        else: #return to host
            return c.get()[0], s.get()[0]

    #TODO cublas_rotm
    #TODO cublas_rotmg
    #TODO cublas_scal
    #TODO cublas_swap

    ## cuBLAS Level-2 Functions ##

    #TODO cublas_gbmv

    #cublas_gemv
    def gemv(self, alpha, A, x, y, beta, op = 'N', incx = 1, incy = 1):
        '''
        y = gemv(self, alpha, A, x, y, beta, op = 'N', incx = 1, incy = 1):
        This function performs the matrix-vector multiplication
        
        y = alpha op(A).x + beta y
        
        where A is a matrix, x and y are vectors and alpha and beta are scalars
        Also, for matrix A

        op(A) = A    if op = 'N'
                A.T  if op = 'T' (transpose)
                A.CT if op = 'C' (complex transpose)

        for op(A) with dimensions m rows x n columns
        x must have dimension n*incx and
        y must have dimension m*incy
        '''
        shape_op = 1 if op=='N' else -1
        (op,) = self._getOPs(op)

        y, x, A, alpha, beta = self._AutoCaster(y, x, A, alpha, beta)

        m,n = A.shape
        lda = m
        if (y.size*incy, x.size*incx)[::shape_op] != (m,n):
            raise ValueError('Matrix and vectors have incompatible dimensions')

        if any([_isOnGPU(alpha), _isOnGPU(beta)]):
            self.pointerMode = 'DEVICE'
            alpha = _toGPU(alpha, y.dtype)
            beta  = _toGPU(beta, y.dtype)
        else:
            self.pointerMode = 'HOST'
            alpha = _ndarray_ptr(alpha)
            beta  = _ndarray_ptr(beta)
        gemv_function = {'float32'    : pycublas.cublasSgemv,
                         'float64'    : pycublas.cublasDgemv,
                         'complex64'  : pycublas.cublasCgemv,
                         'complex128' : pycublas.cublasZgemv
                         }[y.dtype.name]
        self.cublasStatus = gemv_function(self._handle, op.value,
                                          m, n,
                                          alpha.ptr,
                                          A.ptr, lda,
                                          x.ptr, incx,
                                          beta.ptr,
                                          y.ptr, incy)
        return self._return(y)
    
    #TODO leve-2 functions
    
    ## cuBLAS Level-3 Functions ##

    #cublas_gemm
    def gemm(self, alpha, A, B, beta, C, opA = 'N', opB = 'N'):
        '''
        C = gemm(self, alpha, A, B, beta, C, opA = 'N', opB = 'N'):
        This function performs the matrix-matrix multiplication
        
        C = alpha opA(A) opA(B) + beta C
        
        where alpha and beta are scalars, and A, B and C are matrices.
        Also, for matrix X = A or B

        opX(X) = X    if opX = 'N'
                 X.T  if opX = 'T' (transpose)
                 X.CT if opX = 'C' (complex transpose)

        dimensions must be compatible
        opA(A) with m rows x k columns
        opB(B) with k rows x n columns
        C      with m rows x n columns

        If beta = 0 then C does not need to contain valid values.
        '''
        (shape_opA, shape_opB) = [1 if x=='N' else -1 for x in [opA, opB]]
        opA, opB = self._getOPs(opA, opB)
        C, A, B, alpha, beta = self._AutoCaster(C, A, B, alpha, beta)

        m , k = A.shape[::shape_opA]
        kB, n = B.shape[::shape_opB]
        if any([C.shape != (m,n), k != kB]):
            raise ValueError('The matrices have incompatible dimensions')
        
        if any([_isOnGPU(alpha), _isOnGPU(beta)]):
            self.pointerMode = 'DEVICE'
            alpha = _toGPU(alpha, C.dtype)
            beta  = _toGPU(beta, C.dtype)
        else:
            self.pointerMode = 'HOST'
            alpha = _ndarray_ptr(alpha)
            beta  = _ndarray_ptr(beta)
        gemm_function = {'float32'    : pycublas.cublasSgemm,
                         'float64'    : pycublas.cublasDgemm,
                         'complex64'  : pycublas.cublasCgemm,
                         'complex128' : pycublas.cublasZgemm
                         }[C.dtype.name]
        self.cublasStatus = gemm_function(self._handle, 
                                          opA.value, opB.value,
                                          m, n, k,
                                          alpha.ptr,
                                          A.ptr, A.shape[0],
                                          B.ptr, B.shape[0],
                                          beta.ptr,
                                          C.ptr, m)        
        return self._return(C)

    #TODO cublas_gemmBatched
    
    #cublas_symm
    def symm(self, alpha, A, B, beta, C, side = 'L', fillMode = 'U'):
        '''
        This function performs the symmetric matrix-matrix multiplication

        C = alpha A B + beta C    if side = 'L' (left) or
        C = alpha B A + beta C    if side = 'R' (rigth)
        
        where 
        A is a symmetric matrix stored in upper/lower mode (fillMode = 'U'/'L')
        B and C are m rows x n columns matrices, and alpha and beta are scalars
        
        The dimension of A must be
           m x m  if side = 'L' and
           n x n  if side = 'R'
           
        Only the selected triangular part of A (upper or lower) will be used.
        If beta = 0 then C does not need to contain valid values.
        '''
        shape_side = 0 if side=='L' else 1
        (side,) = self._getSIDESs(side)
        (fillMode,) = self._getFILL_MODEs(fillMode)
        C, alpha, A, B, beta = self._AutoCaster(C, alpha, A, B, beta)
        
        m,n  = C.shape
        lda = (m, n)[shape_side]
        if any([C.shape != B.shape, A.shape != (lda, lda)]):
            raise ValueError('The matrices have incompatible dimensions')

        if any([_isOnGPU(alpha), _isOnGPU(beta)]):
            self.pointerMode = 'DEVICE'
            alpha = _toGPU(alpha, C.dtype)
            beta  = _toGPU(beta, C.dtype)
        else:
            self.pointerMode = 'HOST'
            alpha = _ndarray_ptr(alpha)
            beta  = _ndarray_ptr(beta)
        symm_function = {'float32'    : pycublas.cublasSsymm,
                         'float64'    : pycublas.cublasDsymm,
                         'complex64'  : pycublas.cublasCsymm,
                         'complex128' : pycublas.cublasZsymm
                         }[C.dtype.name]
        self.cublasStatus = symm_function(self._handle, 
                                          side.value, fillMode.value,
                                          m, n,
                                          alpha.ptr,
                                          A.ptr, lda,
                                          B.ptr, m,
                                          beta.ptr,
                                          C.ptr, m)
        return self._return(C)

    #cublas_syrk
    def syrk(self, alpha, A, beta, C, fillMode = 'U', op = 'N'):
        '''
        This function performs the symmetric rank-k update

        C = alpha op(A) op(A).T + beta C

        where 
        alpha and beta are scalars, C is a n x n symmetric matrix
        stored in upper/lower mode (fillMode = 'U'/'L')
        and op(A) is a matrix with dimensions n rows x k columns

        op(A) = A    if op = 'N'
                A.T  if op = 'T' (transpose)

        Only the selected triangular part of C (upper or lower) will be used.
        If beta = 0 then C does not need to contain valid values.
        '''
        shape_op = 1 if op=='N' else -1
        (op,) = self._getOPs(op, valid = ['N', 'T'])
        (fillMode,) = self._getFILL_MODEs(fillMode)
        C, alpha, A, beta = self._AutoCaster(C, alpha, A, beta)

        n, k = A.shape[::shape_op]
        if C.shape != (n,n):
            raise ValueError('The matrices have incompatible dimensions')

        if any([_isOnGPU(alpha), _isOnGPU(beta)]):
            self.pointerMode = 'DEVICE'
            alpha = _toGPU(alpha, C.dtype)
            beta  = _toGPU(beta, C.dtype)
        else:
            self.pointerMode = 'HOST'
            alpha = _ndarray_ptr(alpha)
            beta  = _ndarray_ptr(beta)
        syrk_function = {'float32'    : pycublas.cublasSsyrk,
                         'float64'    : pycublas.cublasDsyrk,
                         'complex64'  : pycublas.cublasCsyrk,
                         'complex128' : pycublas.cublasZsyrk
                         }[C.dtype.name]
        self.cublasStatus = syrk_function(self._handle,
                                          fillMode.value, op.value,
                                          n, k,
                                          alpha.ptr,
                                          A.ptr, A.shape[0],
                                          beta.ptr,
                                          C.ptr, n)
        return self._return(C)
