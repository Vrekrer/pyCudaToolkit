# -*- coding: utf-8 -*-

import pycuda.autoinit
import numpy_cublas
import numpy

cublas = numpy_cublas.pycublasContext()

def test_cublasI_amax_amin(vector):
    sinlge_vector = vector.real.astype('float32')
    double_vector = vector.real.copy()
    complex_vector = vector.astype('complex64')
    doubleComplex_vector = vector.copy()

    R_argmax = numpy.argmax(vector.real)
    C_argmax = numpy.argmax(numpy.abs(vector))

    print('I_amax')
    print('dtype , numpy.argmax, cublasI_amax')
    print('float32   ', R_argmax, cublas.cublasI_amax(sinlge_vector))
    print('float64   ', R_argmax, cublas.cublasI_amax(double_vector))
    print('complex64 ', C_argmax, cublas.cublasI_amax(complex_vector))
    print('complex128', C_argmax, cublas.cublasI_amax(doubleComplex_vector))
    print('*****************************')

    R_argmin = numpy.argmin(vector.real)
    C_argmin = numpy.argmin(numpy.abs(vector))

    print('I_amin')
    print('dtype , numpy.argmin, cublasI_amin')
    print('float32   ', R_argmin, cublas.cublasI_amin(sinlge_vector))
    print('float64   ', R_argmin, cublas.cublasI_amin(double_vector))
    print('complex64 ', C_argmin, cublas.cublasI_amin(complex_vector))
    print('complex128', C_argmin, cublas.cublasI_amin(doubleComplex_vector))
    print('*****************************')

