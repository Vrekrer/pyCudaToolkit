# -*- coding: utf-8 -*-

import pycuda.autoinit
import numpy_cublas
import numpy

cublas = numpy_cublas.pycublasContext()

def cublas_1vector_tests(vector):
    sinlge_vector = vector.real.astype('float32')
    double_vector = vector.real.astype('float64')
    complex_vector = vector.astype('complex64')
    doubleComplex_vector = vector.astype('complex128')

    R_argmax = numpy.abs(vector.real).argmax()
    C_argmax = (numpy.abs(vector.real) + numpy.abs(vector.imag)).argmax()

    print('I_amax')
    print('dtype , numpy.abs(A).argmax*, cublasI_amax')
    print('float32   ', R_argmax, cublas.I_amax(sinlge_vector))
    print('float64   ', R_argmax, cublas.I_amax(double_vector))
    print('complex64 ', C_argmax, cublas.I_amax(complex_vector))
    print('complex128', C_argmax, cublas.I_amax(doubleComplex_vector))
    print('* For complex the result is:  (|A.real| + |A.imag|).argmax()')
    print('*****************************')

    R_argmin = numpy.abs(vector.real).argmin()
    C_argmin = (numpy.abs(vector.real) + numpy.abs(vector.imag)).argmin()

    print('I_amin')
    print('dtype , numpy.abs(A).argmin*, cublasI_amin')
    print('float32   ', R_argmin, cublas.I_amin(sinlge_vector))
    print('float64   ', R_argmin, cublas.I_amin(double_vector))
    print('complex64 ', C_argmin, cublas.I_amin(complex_vector))
    print('complex128', C_argmin, cublas.I_amin(doubleComplex_vector))
    print('* For complex the result is:  (|A.real| + |A.imag|).argmin()')
    print('*****************************')

    R_asum = numpy.abs(vector.real).sum()
    C_asum = (numpy.abs(vector.real) + numpy.abs(vector.imag)).sum()

    print('I_asum')
    print('dtype , numpy.abs(A).sum*, cublas_asum')
    print('float32   ', R_asum, cublas.asum(sinlge_vector))
    print('float64   ', R_asum, cublas.asum(double_vector))
    print('complex64 ', C_asum, cublas.asum(complex_vector))
    print('complex128', C_asum, cublas.asum(doubleComplex_vector))
    print('* For complex the result is:  (|A.real| + |A.imag|).sum()')
    print('*****************************')
