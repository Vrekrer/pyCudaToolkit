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
    
class _ndarrayPointer(ctypes.c_void_p):
    '''Pointer object for numpy arrays'''
    def __init__(self, ndarray):
        super(ctypes.c_void_p, self).__init__(ndarray.ctypes.data)
        self.data = ndarray #Keep the array alive


###
### cudart Modules ###
###

def _errorHandler(error):
    '''Raises an exception if error != cudaSuccess.
    
    This funcion is called every time a cudart function is used.
    Overwrite this function if needed!
    '''
    if error != cudart.cudaError_t.cudaSuccess:
        raise NameError('cudaError: ' + error.name)



## 1. Device Management ##


def DeviceReset():
    '''
    Destroy all allocations and reset all state 
    on the current device in the current process. 
    '''
    error = cudart.cudaDeviceReset(dev)
    _errorHandler(error)

def GetDevice():
    '''Returns which device is currently being used.'''
    dev = ctypes.c_int()
    error = cudart.cudaGetDevice(dev)
    _errorHandler(error)
    return dev.value

def GetDeviceCount():
    '''Returns the number of compute-capable devices.'''
    count = ctypes.c_int()
    error = cudart.cudaGetDeviceCount(count)
    _errorHandler(error)
    return count.value

def GetDeviceProperties(dev_num):
    '''Returns information about the compute-device'''
    props = cudart.cudaDeviceProp()
    error = cudart.cudaGetDeviceProperties(props, dev_num)
    _errorHandler(error)
    return props

def SetDevice(dev_num):
    '''Set device to be used for GPU executions.'''
    error = cudart.cudaSetDevice(dev_num)
    _errorHandler(error)

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

def Malloc(size, pointer = 'New'):
    '''Allocate memory on the device.
    
    Args:
        size (int)         : Requested allocation size in bytes
        pointer (c_void_p) : pointer object or 'New'

    Returns:
        if pointer == 'New':
          _cudaMemoryPointer: New pointer to allocated device memory
        otherwise
          the same pointer passed as argument
    '''
    if pointer == 'New':
        pointer = _cudaMemoryPointer()
    error = cudart.cudaMalloc(pointer, size)
    _errorHandler(error)
    return pointer
cudaMalloc = Malloc
MemAlloc = Malloc

def MemCopy(dst, src, count, kind):
    '''Copies memory. 
    
    Args:
        dst (c_void_p)        : Destination pointer
        src (c_void_p)        : Source pointer
        count (int)           : Number of bytes to copy
        kind (int or string)  : Type of transfers
                                0, 'HostToHost', 'H2H'
                                1, 'HostToDevice', 'H2D'
                                2, 'DeviceToHost', 'D2H'
                                3, 'DeviceToDevice' 'D2D'
                                4, 'Default'
    '''
    kinds = {0:0, 'HostToHost':0    , 'H2H':0,
             1:1, 'HostToDevice':1  , 'H2D':1,
             2:2, 'DeviceToHost':2  , 'D2H':2, 
             3:3, 'DeviceToDevice':3, 'D2D':3,
             4:4, 'Default':4}
    if kind not in kinds.keys():
        kind = 'Default'
    kind = cudart.cudaMemcpyKind(kinds[kind])
    error = cudart.cudaMemcpy(dst, src, count, kind)
    _errorHandler(error)
Memcpy = MemCopy


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
