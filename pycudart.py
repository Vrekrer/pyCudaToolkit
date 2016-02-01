# -*- coding: utf-8 -*-

"""
Python functions to cudart
For documentation see:
http://docs.nvidia.com/
"""

import ctypes
import numpy
import cudart
import warnings

class _cudaMemoryPointer(ctypes.c_void_p):
    '''Memory pointer for cuda'''
    def __del__(self):
        error = cudart.cudaFree(self)
        print 'cuda ptr deleted'
    @property
    def ptr(self):
        return self.value
    
class _ndarrayPointer(ctypes.c_void_p):
    '''Pointer object for numpy arrays'''
    def __init__(self, ndarray):
        super(self.__class__, self).__init__(ndarray.ctypes.data)
        self.data = ndarray #Keep the array alive
    def get(self):
        #code compatibility with pycuda.gpuarray
        return self.data
    @property
    def ptr(self):
        return self.value
    @property
    def dtype(self):
        return self.data.dtype

###
### cudart Modules ###
###

def _errorHandler(error):
    '''Raises an exception if error != cudaSuccess.
    
    This funcion is called every time a cudart function is used.
    Overwrite this function if needed.
    '''
    if error != cudart.cudaError_t.cudaSuccess:
        raise NameError('cudaError: ' + error.name)

## 1. Device Management ##


## 3. Error Management ##

def GetErrorName(error):
    '''Returns the string representation of an error code enum name.'''
    return cudart.cudaGetErrorName(error)
     
def GetErrorString(error):
    '''Returns the description string for an error code.'''
    return cudart.cudaGetErrorString(error)

def GetLastError():
    '''Returns the last error from a runtime call.'''
    error = cudart.cudaGetLastError()
    return cudart.cudaError_t(error)

def PeekAtLastError():
    '''Returns the last error from a runtime call.'''
    error = cudart.PeekAtLastError()
    return cudart.cudaError_t(error)

## 9. Memory Managment ##

def Free(pointer):
    '''Frees memory on the device.'''
    error = cudart.cudaFree(pointer)
    _errorHandler(error)
cudaFree = Free

def Malloc(size):
    '''Allocate memory on the device.
    
    Args:
        size (int): Requested allocation size in bytes

    Returns:
        _cudaMemoryPointer: Pointer to allocated device memory
    '''
    pointer = _cudaMemoryPointer()
    error = cudart.cudaMalloc(pointer)
    _errorHandler(error)
cudaMalloc = Malloc

## 26. Version Managment ##
    
def DriverVersion():
    '''Returns the CUDA driver version.'''
    version = ctypes.c_int()
    error = cudart.cudaDriverGetVersion(version)
    _errorHandler(error)
    return version.value

def RuntimeVersion():
    '''Returns the CUDA Runtime version.'''
    version = ctypes.c_int()
    error = cudart.cudaRuntimeGetVersion(version)
    _errorHandler(error)
    return version.value