# -*- coding: utf-8 -*-

import pycuda.autoinit
import numpy_cublas
import numpy

set_test_inputs()

cublas = numpy_cublas.pycublasContext()

def test_cublasI_amax(vector):
    R_argmax = numpy.argmax(vector.real)
    C_argmax = numpy.argmax(numpy.abs(vector))
    
    sinlge_vector = vector.real.astype('float32')
    double_vector = vector.real.copy()
    complex_vector = vector.astype('complex64')
    doubleComplex_vector = vector.copy()
    
    print('float32   ', R_argmax, cublas.cublasI_amax(sinlge_vector))
    print('float64   ', R_argmax, cublas.cublasI_amax(double_vector))
    print('complex64 ', C_argmax, cublas.cublasI_amax(complex_vector))
    print('complex128', C_argmax, cublas.cublasI_amax(doubleComplex_vector))

