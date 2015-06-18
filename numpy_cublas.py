# -*- coding: utf-8 -*-

"""
Python functions to cuBLAS
For documentation see:
http://docs.nvidia.com/cuda/cublas/index.htm
"""

import ctypes
import pycublas
import pycuda.gpuarray
import numpy

def _checkArrayType(array):
    '''Check if array is numpy.ndarray or pycuda.gpuarray.GPUArray'''
    if isinstance(array, pycuda.gpuarray.GPUArray):
        pass
    elif isinstance(array, numpy.ndarray):
        array = pycuda.gpuarray.to_gpu( array )
    else:
        raise TypeError("numpy.ndarray or pycuda.gpuarray.GPUArray expected"
                        "got '%s'" % type(array).__name__)

_valid_GPU_types = ['float32','float64','complex64','complex128']

def _isOnGPU(array):
    return isinstance(array, pycuda.gpuarray.GPUArray)

def _toGPU(array, dtype='Auto'):
    if _isOnGPU(array)
        return array
    elif isinstance(array, numpy.ndarray):
        if dtype == 'Auto':
            dtype = array.dtype.name
        if isinstance(dtype.dtype, numpy.dtype)
            dtype = dtype.dtype.name
        if dtype not in _valid_GPU_types:
            dtype = 'float64'
        return pycuda.gpuarray.to_gpu( array.astype(dtype, copy = True) )
    else: #scalar
        if dtype == 'Auto':
            dtype = None
        return pycuda.gpuarray.to_gpu( numpy.array([array], dtype=dtype) )      

class pycublasContext(object):
    def __init__(self):
        self._handle = pycublas.cublasHandle_t()
        self._cublasStatus = pycublas.cublasCreate(self._handle)
        
        self.CheckStatusFunction = None
    def __del__(self):
        self.cublasStatus = pycublas.cublasDestroy(self._handle)
    
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
    def I_amax(self, array, incx = 1):
        _checkArrayType(array)
        array = _toGPU(array)

        result = ctypes.c_int()        
        I_amax_function = {'float32'    : pycublas.cublasIsamax,
                           'float64'    : pycublas.cublasIdamax,
                           'complex64'  : pycublas.cublasIcamax,
                           'complex128' : pycublas.cublasIzamax
                           }[array.dtype.name]
        
        self.cublasStatus = I_amax_function(self._handle, array.size,
                                            int(array.gpudata), incx, result)
        return result.value - 1        

    # cublasI_amin        
    def I_amin(self, array, incx = 1):
        _checkArrayType(array)
        array = _toGPU(array)
            
        result = ctypes.c_int()        
        I_amin_function = {'float32'    : pycublas.cublasIsamin,
                           'float64'    : pycublas.cublasIdamin,
                           'complex64'  : pycublas.cublasIcamin,
                           'complex128' : pycublas.cublasIzamin
                           }[array.dtype.name]
        
        self.cublasStatus = I_amin_function(self._handle, array.size,
                                            int(array.gpudata), incx, result)
        return result.value - 1  

    # cublas_asum         
    def asum(self, array, incx = 1):
        _checkArrayType(array)
        array = _toGPU(array)
            
        result = ctypes.c_int()        
        asum_function = {'float32'    : pycublas.cublasSasum, 
                         'float64'    : pycublas.cublasDasum,
                         'complex64'  : pycublas.cublasScasum,
                         'complex128' : pycublas.cublasDzasum
                         }[array.dtype.name]
        result_type = {'float32'    : ctypes.c_float,
                       'float64'    : ctypes.c_double,
                       'complex64'  : ctypes.c_float,
                       'complex128' : ctypes.c_double
                       }[array.dtype.name]   
                         
        result = result_type()
        self.cublasStatus = asum_function(self._handle, array.size,
                                          int(array.gpudata), incx, result)
        return result.value
        
    # cublas_axpy         
    def axpy(self, alpha, X, Y, incx = 1, incy = 1):
        '''
        Y = alpha * X + Y
        '''
        _checkArrayType(X)
        _checkArrayType(Y)

        Ygpu = _toGPU(Y, dtype=Y.dtype)
        X = _toGPU(X, dtype=Y.dtype)

        #TODO Allow HOST scalars
        self.pointerMode = 'DEVICE'
        alpha = _toGPU(alpha, dtype=Y.dtype)
            
        axpy_function = {'float32'    : pycublas.cublasSaxpy, 
                         'float64'    : pycublas.cublasDaxpy,
                         'complex64'  : pycublas.cublasCaxpy,
                         'complex128' : pycublas.cublasZaxpy
                         }[Ygpu.dtype.name]
                         
        self.cublasStatus = axpy_function(self._handle, Ygpu.size,
                                          alpha.ptr,
                                          X.ptr, incx,
                                          Ygpu.ptr, incy)
                                          
        #fill original Y if needed (get the data from gpu)
        if isinstance(Y, numpy.ndarray):
            Ygpu.get(Y)
            

        
        
        
