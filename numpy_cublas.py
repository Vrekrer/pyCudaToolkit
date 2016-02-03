# -*- coding: utf-8 -*-

"""
Python functions to cuBLAS

For documentation see:
http://docs.nvidia.com/cuda/cublas/
"""

import ctypes
import numpy
import cublas
import pycudart
import numbers

#pycuda classes
try:
    import pycudax.gpuarray
    pycudaGPUArray = pycuda.gpuarray.GPUArray
    PYCUDA_AVIALABLE = True
except:
    pycudaGPUArray = type('no_pycuda', (), {})
    PYCUDA_AVIALABLE = False

_valid_array_dtypes = ['float32','float64','complex64','complex128']

class cublasData(ctypes.c_void_p ,object):
    '''Pointers used for cublas
    '''
    def __init__(self, data, dtype='Auto', memType = 'device'):
        '''Creates a  memory pointer to a scalar or an array
        
        Args:
            value   : Value of the scalar or ndarray
            dtype   : 'float32', 'float64', 'complex64', 'complex128' or 'Auto'
            memType : 'host' or 'device' 
        '''
        if not isinstance(data, (numpy.ndarray, numbers.Number)):
            raise TypeError("value must be a number or a numpy array")
        if memType not in ['host', 'device']:
            raise ValueError("memType must be 'host' or 'device'")
            
        _ndarray = numpy.atleast_1d(data)
        if dtype == 'Auto':
            dtype = _ndarray.dtype
        if numpy.dtype(dtype).name not in _valid_array_dtypes:
            dtype = 'float64'
        _ndarray = _ndarray.astype(dtype, order='F', copy=False)
        
        self._dtype  = _ndarray.dtype
        self._shape  = _ndarray.shape
        self._flags  = _ndarray.flags
        self._size   = _ndarray.size
        self._nbytes = _ndarray.nbytes
        
        self._memType = memType
        if self._memType == 'host':
            self._ndarray = _ndarray
            super(ctypes.c_void_p, self).__init__(self._ndarray.ctypes.data)
        elif self._memType == 'device':
            super(ctypes.c_void_p, self).__init__(None)
            pycudart.MemAlloc(self._nbytes, self)
            pycudart.MemCopy(self, _ndarray.ctypes.data, 
                             self._nbytes, 'HostToDevice')
            
    def __del__(self):
        self.Free()
        
    def __repr__(self):
        return 'cublasData @%s pointing to %s at %s' %(id(self), 
                                                       self.value, 
                                                       self._memType)
        
    def Free(self):
        '''
        Frees the memory in the device or 
        deletes the array reference in the host
        '''
        if self._memType == 'host':
            del self._ndarray
        elif self._memType == 'device':
            pycudart.Free(self)
        super(ctypes.c_void_p, self).__init__(None)
    
    def astype(self, dtype = 'no change', memType = 'no change'):
        if dtype == 'no change':
            dtype = self._dtype
        if memType == 'no change':
            memType = self._memType
        if numpy.dtype(dtype).name not in _valid_array_dtypes:
            raise ValueError("Invalid dtype")
            
        if (numpy.dtype(dtype) != self._dtype) or (memType != self._memType):
            return cublasData(self.data, dtype, memType)
        else:
            return self
    
    def copy(self, dtype = 'no change', memType = 'no change'):
        if (dtype != 'no change') or (memType != 'no change'):
            return self.astype(dtype, memType)
        
        newCopy = ctypes._SimpleCData.__new__(cublasData)
        newCopy.__dict__ = self.__dict__.copy()
        if self._memType == 'host':
            newCopy._ndarray = self._ndarray.copy(order = 'F')
            super(ctypes.c_void_p, newCopy).__init__(newCopy._ndarray.ctypes.data)
        elif self._memType == 'device':
            super(ctypes.c_void_p, newCopy).__init__(None)
            pycudart.MemAlloc(newCopy._nbytes, newCopy)
            pycudart.MemCopy(newCopy, self, 
                             newCopy._nbytes, 'DeviceToDevice')
        return newCopy
            
    @property
    def flags(self):
        return self._flags
    @property
    def dtype(self):
        return self._dtype
    @property
    def shape(self):
        return self._shape
    @property
    def size(self):
        return self._size
    @property
    def nbytes(self):
        return self._nbytes
        
    @property
    def memType(self):
        return self._memType
    @property
    def isOnGPU(self):
        return self._memType == 'device'
    @property
    def isOnCPU(self):
        return self._memType == 'host'
        
        
    @property
    def data(self):
        if self._memType == 'host':
            data = self._ndarray
        elif self._memType == 'device':
            data = numpy.zeros(self._shape, dtype=self._dtype, order='F')
            pycudart.MemCopy(data.ctypes.data, self, 
                             data.nbytes, 'DeviceToHost')

        if data.size == 1:
            return data.item()
        else:
            return data

class cublasContext(object):
    def __init__(self):
        self._handle = cublas.cublasHandle_t()
        self._cublasStatus = cublas.cublasCreate(self._handle)
        
        self.CheckStatusFunction = None
        
        self._returnToHost = True
        
    def __del__(self):
        self.cublasStatus = cublas.cublasDestroy(self._handle)

    def _AutoCaster(self, *args):
        '''
        Function used for automatic dtypes casting of args
        
        Args: list of numbers, numpy arrays, 
              cublasData objects or pycuda GPUArrays
        
        Returns a list of cublasData pointers of the args.
        Largest dtype is calculated and used to cast numbers and arrays, 
        cublasData objects are recasted if needed
        Numbers are returned as host pointers while arrays are loaded 
        into device memory
        '''
        _areComplex = any( [isinstance(x, complex)
                            if isinstance(x, numbers.Number)
                            else ('complex' in x.dtype.name)
                            for x in args] )
        _areSingle = all( [True if isinstance(x, numbers.Number)
                           else x.dtype.name not in ['float64','complex128']
                           for x in args] )
        dtype = {(False,True ):'float32',
                 (False,False):'float64',
                 (True, True ):'complex64',
                 (True, False):'complex128'}[(_areComplex, _areSingle)]
        
        return [cublasData(x, dtype, 'host') if isinstance(x, numbers.Number)
                else x.astype(dtype, 'device') if isinstance(x, cublasData)
                else cublasData(x, dtype, 'device')
                for x in args]

    def _getOPs(self, *args, **kargs):
        '''
        Check OPERATION strings and return them as 
        valid cublas.cublasOperation_t instances
        '''
        op_dict = {'N': cublas.cublasOperation_t.CUBLAS_OP_N,
                   'T': cublas.cublasOperation_t.CUBLAS_OP_T,
                   'C': cublas.cublasOperation_t.CUBLAS_OP_C}
        valid_keys = kargs['valid'] if 'valid' in kargs else op_dict.keys()
        if all([op in valid_keys for op in args]):
            return (op_dict[op] for op in args)
        else:
            valid_string = ', '.join("'%s'"% s for s in valid_keys[:-1]) + \
                           " or '%s'" % valid_keys[-1]
            raise ValueError("op must be %s" % valid_string)

    def _getSIDESs(self, *args):
        '''
        Check SIDE strings and return them as 
        valid cublas.cublasSideMode_t instances
        '''
        side_dict = {'L': cublas.cublasSideMode_t.CUBLAS_SIDE_LEFT,
                     'R': cublas.cublasSideMode_t.CUBLAS_SIDE_RIGHT}
        if all([side in side_dict.keys() for side in args]):
            return (side_dict[side] for side in args)
        else:
            raise ValueError("side must be 'L' (left) or 'R' (rigth)")

    def _getFILL_MODEs(self, *args):
        '''
        Check FILL_MODE strings and return them as 
        valid cublas.cublasFillMode_t instances
        '''
        fm_dict = {'U': cublas.cublasFillMode_t.CUBLAS_FILL_MODE_UPPER,
                   'L': cublas.cublasFillMode_t.CUBLAS_FILL_MODE_LOWER}
        if all([fm in fm_dict.keys() for fm in args]):
            return (fm_dict[fm] for fm in args)
        else:
            raise ValueError("fillMode must be 'U' (upper) or 'L' (lower)")

    def _getDIAGs(self, *args):
        '''
        Check DIAGONAL strings and return them as 
        valid cublas.cublasDiagType_t instances
        '''
        diag_dict = {'N': cublas.cublasDiagType_t.CUBLAS_DIAG_NON_UNIT,
                     'U': cublas.cublasDiagType_t.CUBLAS_DIAG_UNIT}
        if all([diag in diag_dict.keys() for diag in args]):
            return (diag_dict[diag] for diag in args)
        else:
            raise ValueError("diag must be 'U' (unit) or 'N' (non unit)")

    @property
    def returnToHost(self):
        '''
        If True return arrays are instances of numpy.ndarray
        returnToHost = not(returnToDevice)
        '''
        return self._returnToHost
    @returnToHost.setter
    def returnToHost(self, value):
        self._returnToHost = bool(value)
    @property
    def returnToDevice(self):
        '''
        If True return arrays are instances of pycuda.gpuarray.GPUArray
        returnToDevice = not(returnToHost)
        '''
        return not self._returnToHost
    @returnToDevice.setter
    def returnToDevice(self, value):
        self._returnToHost = not bool(value)

    def _return(self, pointer):
        if self.returnToDevice:
            return pointer
        elif self.returnToHost:
            return pointer.data
    
    ## cublasStatus Check ##
    @property
    def cublasStatus(self):
        return self._cublasStatus
    @cublasStatus.setter
    def cublasStatus(self, status):
        if isinstance(status, cublas.cublasStatus_t):
            self._cublasStatus = status
        if callable(self.CheckStatusFunction):
            self.CheckStatusFunction(self._cublasStatus)
        
    ## cuBLAS Helper Functions ##
    @property
    def Version(self):
        version = ctypes.c_int()
        self.cublasStatus = cublas.cublasGetVersion(self._handle, version)
        return version.value

    # cublasPointerMode
    @property
    def pointerMode(self):
        '''
        Indicates whether the scalar values are passed 
        by reference on the host or device
        
        This property is set to the appropiate value on each function call
        '''
        pMode = cublas.c_cublasPointerMode_t()
        self.cublasStatus = cublas.cublasGetPointerMode(self._handle, pMode)
        return cublas.cublasPointerMode_t(pMode.value)
    @pointerMode.setter
    def pointerMode(self, mode):
        if isinstance(mode, cublas.cublasPointerMode_t):
            mode = mode.value
        if mode in ['CUBLAS_POINTER_MODE_HOST', 0, 
                    'Host', 'HOST', 'host']:
            mode = 0
        elif mode in ['CUBLAS_POINTER_MODE_DEVICE', 1, 
                      'Device', 'DEVICE', 'device']:
            mode = 1
        else:
            mode = self.pointerMode.value
        self.cublasStatus = cublas.cublasSetPointerMode(self._handle, mode)

    # cublasAtomicsMode       
    @property
    def atomicsMode(self):
        '''
        Indicates whether cuBLAS routines which has an 
        alternate implementation using atomics can be used
        '''
        aMode = cublas.c_cublasAtomicsMode_t()
        self.cublasStatus = cublas.cublasGetAtomicsMode(self._handle, aMode)
        return cublas.cublasAtomicsMode_t(aMode.value)
    @atomicsMode.setter
    def atomicsMode(self, mode):
        if isinstance(mode, cublas.cublasAtomicsMode_t):
            mode = mode.value
        if mode in ['CUBLAS_ATOMICS_NOT_ALLOWED', 0, False, 'NOT_ALLOWED']:
            mode = 0
        elif mode in ['CUBLAS_ATOMICS_ALLOWED', 1, True, 'ALLOWED']:
            mode = 1
        else:
            mode = self.atomicsMode.value
        self.cublasStatus = cublas.cublasSetAtomicsMode(self._handle, mode)

    ## cuBLAS Level-1 Functions ##
    
    # cublasI_amax
    def I_amax(self, X, incx = 1):
        X = self._AutoCaster(X)[0]
    
        I_amax_function = {'float32'    : cublas.cublasIsamax,
                           'float64'    : cublas.cublasIdamax,
                           'complex64'  : cublas.cublasIcamax,
                           'complex128' : cublas.cublasIzamax
                           }[X.dtype.name]
        result = ctypes.c_int()
        
        self.cublasStatus = I_amax_function(self._handle, X.size,
                                            X, incx, result)
        return result.value - 1        

    # cublasI_amin        
    def I_amin(self, X, incx = 1):
        X = self._AutoCaster(X)[0]
      
        I_amin_function = {'float32'    : cublas.cublasIsamin,
                           'float64'    : cublas.cublasIdamin,
                           'complex64'  : cublas.cublasIcamin,
                           'complex128' : cublas.cublasIzamin
                           }[X.dtype.name]
        result = ctypes.c_int()
        
        self.cublasStatus = I_amin_function(self._handle, X.size,
                                            X, incx, result)
        return result.value - 1  

    # cublas_asum         
    def asum(self, X, incx = 1):
        X = self._AutoCaster(X)[0]
                  
        asum_function = {'float32'    : cublas.cublasSasum, 
                         'float64'    : cublas.cublasDasum,
                         'complex64'  : cublas.cublasScasum,
                         'complex128' : cublas.cublasDzasum
                         }[X.dtype.name]
        result_type = {'float32'    : ctypes.c_float,
                       'float64'    : ctypes.c_double,
                       'complex64'  : ctypes.c_float,
                       'complex128' : ctypes.c_double
                       }[X.dtype.name]   
        result = result_type()
        
        self.cublasStatus = asum_function(self._handle, X.size,
                                          X, incx, result)
        return result.value

    # cublas_axpy         
    def axpy(self, alpha, X, Y, incx = 1, incy = 1):
        '''
        Y = alpha * X + Y
        '''
        Y, alpha, X = self._AutoCaster(Y, alpha, X)

        self.pointerMode = alpha.memType
        axpy_function = {'float32'    : cublas.cublasSaxpy, 
                         'float64'    : cublas.cublasDaxpy,
                         'complex64'  : cublas.cublasCaxpy,
                         'complex128' : cublas.cublasZaxpy
                         }[Y.dtype.name]
        self.cublasStatus = axpy_function(self._handle, Y.size,
                                          alpha,
                                          X, incx,
                                          Y, incy)
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
            dot_function = {'float32' : cublas.cublasSdot, 
                            'float64' : cublas.cublasDdot
                           }[Y.dtype.name]
        else: # complex
            dot_function = {('complex64' , False) : cublas.cublasCdotu,
                            ('complex128', False) : cublas.cublasZdotu,
                            ('complex64' , True)  : cublas.cublasCdotc,
                            ('complex128', True)  : cublas.cublasZdotc,
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
                  
        nrm2_function = {'float32'    : cublas.cublasSnrm2, 
                         'float64'    : cublas.cublasDnrm2,
                         'complex64'  : cublas.cublasScnrm2,
                         'complex128' : cublas.cublasDznrm2
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
            dot_function = {'float32' : cublas.cublasSrot,
                            'float64' : cublas.cublasDrot
                           }[Y.dtype.name]
        else: # complex
            s_complex = (s[0].imag != 0)
            dot_function = {('complex64' , True) : cublas.cublasCrot,
                            ('complex128', True) : cublas.cublasZrot,
                            ('complex64' , False): cublas.cublasCsrot,
                            ('complex128', False): cublas.cublasZdrot,
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
        rotg_function = {'float32'    : cublas.cublasSrotg,
                         'float64'    : cublas.cublasDrotg,
                         'complex64'  : cublas.cublasCrotg,
                         'complex128' : cublas.cublasZrotg
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
        op_type = 1.0j if op == 'C' else 1
        (op,) = self._getOPs(op)

        y, x, A, alpha, beta, op_type = self._AutoCaster(y, x, A, alpha, beta,
                                                         op_type)

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
        gemv_function = {'float32'    : cublas.cublasSgemv,
                         'float64'    : cublas.cublasDgemv,
                         'complex64'  : cublas.cublasCgemv,
                         'complex128' : cublas.cublasZgemv
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
        op_type = 1.0j if 'C' in [opA, opB] else 1
        opA, opB = self._getOPs(opA, opB)
        C, A, B, alpha, beta, op_type = self._AutoCaster(C, A, B, alpha, beta,
                                                         op_type)

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
        gemm_function = {'float32'    : cublas.cublasSgemm,
                         'float64'    : cublas.cublasDgemm,
                         'complex64'  : cublas.cublasCgemm,
                         'complex128' : cublas.cublasZgemm
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
        symm_function = {'float32'    : cublas.cublasSsymm,
                         'float64'    : cublas.cublasDsymm,
                         'complex64'  : cublas.cublasCsymm,
                         'complex128' : cublas.cublasZsymm
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

        Only the selected triangular part of C (upper or lower)
        will be used and returned.
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
        syrk_function = {'float32'    : cublas.cublasSsyrk,
                         'float64'    : cublas.cublasDsyrk,
                         'complex64'  : cublas.cublasCsyrk,
                         'complex128' : cublas.cublasZsyrk
                         }[C.dtype.name]
        self.cublasStatus = syrk_function(self._handle,
                                          fillMode.value, op.value,
                                          n, k,
                                          alpha.ptr,
                                          A.ptr, A.shape[0],
                                          beta.ptr,
                                          C.ptr, n)
        return self._return(C)

    #cublas_syr2k
    def syr2k(self, alpha, A, B, beta, C, fillMode = 'U', op = 'N'):
        '''
        This function performs the symmetric rank-2k update

        C = alpha ( op(A) op(B).T + op(B) op(A).T ) + beta C

        where
        alpha and beta are scalars, C is a n x n symmetric matrix
        stored in upper/lower mode (fillMode = 'U'/'L')
        and op(A) and op(B) are matrices with dimensions n rows x k columns

        for X = A or B
        op(X) = X    if op = 'N'
                X.T  if op = 'T' (transpose)

        Only the selected triangular part of C (upper or lower)
        will be used and returned.
        If beta = 0 then C does not need to contain valid values.
        '''
        shape_op = 1 if op=='N' else -1
        (op,) = self._getOPs(op, valid = ['N', 'T'])
        (fillMode,) = self._getFILL_MODEs(fillMode)
        C, alpha, A, B, beta = self._AutoCaster(C, alpha, A, B, beta)

        n, k = A.shape[::shape_op]
        if any([C.shape != (n,n), B.shape != A.shape]):
            raise ValueError('The matrices have incompatible dimensions')

        if any([_isOnGPU(alpha), _isOnGPU(beta)]):
            self.pointerMode = 'DEVICE'
            alpha = _toGPU(alpha, C.dtype)
            beta  = _toGPU(beta, C.dtype)
        else:
            self.pointerMode = 'HOST'
            alpha = _ndarray_ptr(alpha)
            beta  = _ndarray_ptr(beta)
        syr2k_function = {'float32'    : cublas.cublasSsyr2k,
                          'float64'    : cublas.cublasDsyr2k,
                          'complex64'  : cublas.cublasCsyr2k,
                          'complex128' : cublas.cublasZsyr2k
                          }[C.dtype.name]
        self.cublasStatus = syr2k_function(self._handle,
                                           fillMode.value, op.value,
                                           n, k,
                                           alpha.ptr,
                                           A.ptr, A.shape[0],
                                           B.ptr, B.shape[0],
                                           beta.ptr,
                                           C.ptr, n)
        return self._return(C)

    #cublas_syrkx
    def syrkx(self, alpha, A, B, beta, C, fillMode = 'U', op = 'N'):
        '''
        This function performs a variation of the symmetric rank-k update

        C = alpha op(A) op(B).T + beta C

        where
        alpha and beta are scalars, C is a n x n symmetric matrix
        stored in upper/lower mode (fillMode = 'U'/'L')
        and op(A) and op(B) are matrices with dimensions n rows x k columns

        for X = A or B
        op(X) = X    if op = 'N'
                X.T  if op = 'T' (transpose)

        Only the selected triangular part of C (upper or lower)
        will be used and returned.
        If beta = 0 then C does not need to contain valid values.
        
        This routine can be used when B is in such way that the result 
        is garanteed to be symmetric. An usual example is when the 
        matrix B is a scaled form of the matrix A : this is equivalent to
        B being the product of the matrix A and a diagonal matrix. 
        For an efficient computation of the product of a regular matrix 
        with a diagonal matrix, refer to the routine dgmm.
        '''
        shape_op = 1 if op=='N' else -1
        (op,) = self._getOPs(op, valid = ['N', 'T'])
        (fillMode,) = self._getFILL_MODEs(fillMode)
        C, alpha, A, B, beta = self._AutoCaster(C, alpha, A, B, beta)

        n, k = A.shape[::shape_op]
        if any([C.shape != (n,n), B.shape != A.shape]):
            raise ValueError('The matrices have incompatible dimensions')

        if any([_isOnGPU(alpha), _isOnGPU(beta)]):
            self.pointerMode = 'DEVICE'
            alpha = _toGPU(alpha, C.dtype)
            beta  = _toGPU(beta, C.dtype)
        else:
            self.pointerMode = 'HOST'
            alpha = _ndarray_ptr(alpha)
            beta  = _ndarray_ptr(beta)
        syrkx_function = {'float32'    : cublas.cublasSsyrkx,
                          'float64'    : cublas.cublasDsyrkx,
                          'complex64'  : cublas.cublasCsyrkx,
                          'complex128' : cublas.cublasZsyrkx
                          }[C.dtype.name]
        self.cublasStatus = syrkx_function(self._handle,
                                           fillMode.value, op.value,
                                           n, k,
                                           alpha.ptr,
                                           A.ptr, A.shape[0],
                                           B.ptr, B.shape[0],
                                           beta.ptr,
                                           C.ptr, n)
        return self._return(C)

    #cublas_trmm
    def trmm(self, alpha, A, B, C, 
             side = 'L', fillMode = 'U', op = 'N', diag = 'N'):
        '''
        This function performs the triangular matrix-matrix multiplication

        C = alpha op(A) B    if side = 'L' (left) or
        C = alpha B op(A)    if side = 'R' (rigth)

        where
        alpha is a scalars, B and C are m rows x n columns matrices and
        A is a triangular matrix stored in upper/lower mode (fillMode = 'U'/'L')
        with or without the main diagonal. Also, for matrix A

        op(A) = A    if op = 'N'
                A.T  if op = 'T' (transpose)
                A.CT if op = 'C' (complex transpose)

        A is assumed to be unit triangular     if diag = 'U'
        A is not assumed to be unit triangular if diag = 'N'

        The dimension of op(A) must be
           m x m  if side = 'L' and
           n x n  if side = 'R'

        Only the selected triangular part of A (upper or lower) will be used.
        C is used only to write the results.
        B can be repeted as C parameter to write the results into B
        '''
        shape_side = 0 if side=='L' else 1
        op_type = 1.0j if op == 'C' else 1
        (op,) = self._getOPs(op)
        (side,) = self._getSIDESs(side)
        (fillMode,) = self._getFILL_MODEs(fillMode)
        (diag,) = self._getDIAGs(diag)
        C, alpha, A, B, op_type = self._AutoCaster(C, alpha, A, B, op_type)

        m, n = B.shape
        lda = B.shape[shape_side]
        if any([C.shape != (m,n), A.shape != (lda,lda)]):
            raise ValueError('The matrices have incompatible dimensions')

        if _isOnGPU(alpha):
            self.pointerMode = 'DEVICE'
        else:
            self.pointerMode = 'HOST'
            alpha = _ndarray_ptr(alpha)
        trmm_function = {'float32'    : cublas.cublasStrmm,
                         'float64'    : cublas.cublasDtrmm,
                         'complex64'  : cublas.cublasCtrmm,
                         'complex128' : cublas.cublasZtrmm
                         }[C.dtype.name]
        self.cublasStatus = trmm_function(self._handle,
                                          side.value, fillMode.value,
                                          op.value, diag.value,
                                          m, n,
                                          alpha.ptr,
                                          A.ptr, lda,
                                          B.ptr, m,
                                          C.ptr, m)
        return self._return(C)

    #cublas_trsm
    def trsm(self, alpha, A, B,
             side = 'L', fillMode = 'U', op = 'N', diag = 'N'):
        '''
        This function solves the triangular linear system with 
        multiple right-hand-sides

        op(A) X = alpha B    if side = 'L' (left) or
        op(A) X = alpha B    if side = 'R' (rigth)

        where
        alpha is a scalar, A is a triangular matrix stored in upper/lower mode 
        (fillMode = 'U'/'L') with or without the main diagonal,
        B and X are m rows x n columns matrices.
        Also, for matrix A

        op(A) = A    if op = 'N'
                A.T  if op = 'T' (transpose)
                A.CT if op = 'C' (complex transpose)

        A is assumed to be unit triangular     if diag = 'U'
        A is not assumed to be unit triangular if diag = 'N'

        The dimension of op(A) must be
           m x m  if side = 'L' and
           n x n  if side = 'R'
           
        Only the selected triangular part of A (upper or lower) will be used.
        The solution X overwrites the right-hand-sides B on exit.
        
        No test for singularity or near-singularity is included in this function.
        '''
        shape_side = 0 if side=='L' else 1
        op_type = 1.0j if op == 'C' else 1
        (op,) = self._getOPs(op)
        (side,) = self._getSIDESs(side)
        (fillMode,) = self._getFILL_MODEs(fillMode)
        (diag,) = self._getDIAGs(diag)
        alpha, A, B, op_type = self._AutoCaster(alpha, A, B, op_type)

        m, n = B.shape
        lda = B.shape[shape_side]
        if A.shape != (lda,lda):
            raise ValueError('The matrices have incompatible dimensions')

        if _isOnGPU(alpha):
            self.pointerMode = 'DEVICE'
        else:
            self.pointerMode = 'HOST'
            alpha = _ndarray_ptr(alpha)
        trsm_function = {'float32'    : cublas.cublasStrsm,
                         'float64'    : cublas.cublasDtrsm,
                         'complex64'  : cublas.cublasCtrsm,
                         'complex128' : cublas.cublasZtrsm
                         }[A.dtype.name]
        self.cublasStatus = trsm_function(self._handle,
                                          side.value, fillMode.value,
                                          op.value, diag.value,
                                          m, n,
                                          alpha.ptr,
                                          A.ptr, lda,
                                          B.ptr, m)
        return self._return(B)
