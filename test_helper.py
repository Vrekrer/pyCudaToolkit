# -*- coding: utf-8 -*-

from cublas import *
from ctypes import *

#Test functions
handle = cublasHandle_t()

def Init():
    status = cublasCreate(byref(handle))
    print status

def Close():
    status = cublasDestroy(handle)
    print status

def Version():
    version = c_int()
    status = cublasGetVersion(handle, byref(version))
    print 'version:', version.value
    print status
  
def GetPointerMode():
    mode = c_cublasPointerMode_t()
    status = cublasGetPointerMode(handle, byref(mode))
    print 'PointerMode:', cublasPointerMode_t(mode.value)
    print status

def SetPointerMode(mode):
    mode = c_cublasPointerMode_t(mode)
    status = cublasSetPointerMode(handle, mode)
    print status
    
def GetAtomicsMode():
    mode = c_cublasAtomicsMode_t()
    status = cublasGetAtomicsMode(handle, byref(mode))
    print 'AtomicMode:', cublasAtomicsMode_t(mode.value)
    print status

def SetAtomicsMode(mode):
    mode = c_cublasAtomicsMode_t(mode)
    status = cublasSetAtomicsMode(handle, mode)
    print status

