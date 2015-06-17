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
        result = ctypes.c_int()
        if not(isinstance(array, pycuda.gpuarray.GPUArray)):
            array = pycuda.gpuarray.to_gpu( numpy.atleast_1d(array) )
        
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
        result = ctypes.c_int()
        if not(isinstance(array, pycuda.gpuarray.GPUArray)):
            array = pycuda.gpuarray.to_gpu( numpy.atleast_1d(array) )
        
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
        if not(isinstance(array, pycuda.gpuarray.GPUArray)):
            array = pycuda.gpuarray.to_gpu( numpy.atleast_1d(array) )
        
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
        
