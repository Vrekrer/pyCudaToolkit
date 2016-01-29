# -*- coding: utf-8 -*-

"""
Raw ctypes wrappers of the cuda runtime api library v7.0+ (libcudart.so.7.0)

For documentation see:
http://docs.nvidia.com/cuda/cuda-runtime-api/
"""

import platform
import ctypes
import ctypes.util
import enum
from ctypes import *

### cudart Library ###
libname = ctypes.util.find_library('cudart')
#TODO import name for windows/mac?
if platform.system()=='Windows': 
    libcudart = ctypes.windll.LoadLibrary(libname)
elif platform.system()=='Linux':     
    libcudart = ctypes.CDLL(libname, ctypes.RTLD_GLOBAL)
else:
    libcudart = ctypes.cdll.LoadLibrary(libname)

### Datatypes ###

##Enumerates##

#cudaError_t
class cudaError_t(enum.IntEnum):
    cudaSuccess = 0
    cudaErrorMissingConfiguration       = 1
    cudaErrorMemoryAllocation           = 2
    cudaErrorInitializationError        = 3
    cudaErrorLaunchFailure              = 4
    cudaErrorPriorLaunchFailure         = 5
    cudaErrorLaunchTimeout              = 6
    cudaErrorLaunchOutOfResources       = 7
    cudaErrorInvalidDeviceFunction      = 8
    cudaErrorInvalidConfiguration       = 9
    cudaErrorInvalidDevice              = 10
    cudaErrorInvalidValue               = 11
    cudaErrorInvalidPitchValue          = 12
    cudaErrorInvalidSymbol              = 13
    cudaErrorMapBufferObjectFailed      = 14
    cudaErrorUnmapBufferObjectFailed    = 15
    cudaErrorInvalidHostPointer         = 16
    cudaErrorInvalidDevicePointer       = 17
    cudaErrorInvalidTexture             = 18
    cudaErrorInvalidTextureBinding      = 19
    cudaErrorInvalidChannelDescriptor   = 20
    cudaErrorInvalidMemcpyDirection     = 21
    cudaErrorAddressOfConstant          = 22
    cudaErrorTextureFetchFailed         = 23
    cudaErrorTextureNotBound            = 24
    cudaErrorSynchronizationError       = 25
    cudaErrorInvalidFilterSetting       = 26
    cudaErrorInvalidNormSetting         = 27
    cudaErrorMixedDeviceExecution       = 28
    cudaErrorCudartUnloading            = 29
    cudaErrorUnknown                    = 30
    cudaErrorNotYetImplemented          = 31
    cudaErrorMemoryValueTooLarge        = 32
    cudaErrorInvalidResourceHandle      = 33
    cudaErrorNotReady                   = 34
    cudaErrorInsufficientDriver         = 35
    cudaErrorSetOnActiveProcess         = 36
    cudaErrorInvalidSurface             = 37
    cudaErrorNoDevice                   = 38
    cudaErrorECCUncorrectable           = 39
    cudaErrorSharedObjectSymbolNotFound = 40
    cudaErrorSharedObjectInitFailed     = 41
    cudaErrorUnsupportedLimit           = 42
    cudaErrorDuplicateVariableName      = 43
    cudaErrorDuplicateTextureName       = 44
    cudaErrorDuplicateSurfaceName       = 45
    cudaErrorDevicesUnavailable         = 46
    cudaErrorInvalidKernelImage         = 47
    cudaErrorNoKernelImageForDevice     = 48
    cudaErrorIncompatibleDriverContext  = 49
    cudaErrorPeerAccessAlreadyEnabled   = 50
    cudaErrorPeerAccessNotEnabled       = 51
    cudaErrorDeviceAlreadyInUse         = 54
    cudaErrorProfilerDisabled           = 55
    cudaErrorProfilerNotInitialized     = 56
    cudaErrorProfilerAlreadyStarted     = 57
    cudaErrorProfilerAlreadyStopped     = 58
    cudaErrorAssert                     = 59
    cudaErrorTooManyPeers               = 60
    cudaErrorHostMemoryAlreadyRegistered = 61
    cudaErrorHostMemoryNotRegistered    = 62
    cudaErrorOperatingSystem            = 63
    cudaErrorPeerAccessUnsupported      = 64
    cudaErrorLaunchMaxDepthExceeded     = 65
    cudaErrorLaunchFileScopedTex        = 66
    cudaErrorLaunchFileScopedSurf       = 67
    cudaErrorSyncDepthExceeded          = 68
    cudaErrorLaunchPendingCountExceeded = 69
    cudaErrorNotPermitted               = 70
    cudaErrorNotSupported               = 71
    cudaErrorHardwareStackError         = 72
    cudaErrorIllegalInstruction         = 73
    cudaErrorMisalignedAddress          = 74
    cudaErrorInvalidAddressSpace        = 75
    cudaErrorInvalidPc                  = 76
    cudaErrorIllegalAddress             = 77
    cudaErrorInvalidPtx                 = 78
    cudaErrorInvalidGraphicsContext     = 79
    cudaErrorStartupFailure             = 0x7f
    cudaErrorApiFailureBase             = 10000
c_cudaError_t = c_int

#cudaDeviceAttr
class cudaDeviceAttr(enum.IntEnum):
    cudaDevAttrMaxThreadsPerBlock             = 1
    cudaDevAttrMaxBlockDimX                   = 2
    cudaDevAttrMaxBlockDimY                   = 3
    cudaDevAttrMaxBlockDimZ                   = 4
    cudaDevAttrMaxGridDimX                    = 5
    cudaDevAttrMaxGridDimY                    = 6
    cudaDevAttrMaxGridDimZ                    = 7
    cudaDevAttrMaxSharedMemoryPerBlock        = 8
    cudaDevAttrTotalConstantMemory            = 9
    cudaDevAttrWarpSize                       = 10
    cudaDevAttrMaxPitch                       = 11
    cudaDevAttrMaxRegistersPerBlock           = 12
    cudaDevAttrClockRate                      = 13
    cudaDevAttrTextureAlignment               = 14
    cudaDevAttrGpuOverlap                     = 15
    cudaDevAttrMultiProcessorCount            = 16
    cudaDevAttrKernelExecTimeout              = 17
    cudaDevAttrIntegrated                     = 18
    cudaDevAttrCanMapHostMemory               = 19
    cudaDevAttrComputeMode                    = 20
    cudaDevAttrMaxTexture1DWidth              = 21
    cudaDevAttrMaxTexture2DWidth              = 22
    cudaDevAttrMaxTexture2DHeight             = 23
    cudaDevAttrMaxTexture3DWidth              = 24
    cudaDevAttrMaxTexture3DHeight             = 25
    cudaDevAttrMaxTexture3DDepth              = 26
    cudaDevAttrMaxTexture2DLayeredWidth       = 27
    cudaDevAttrMaxTexture2DLayeredHeight      = 28
    cudaDevAttrMaxTexture2DLayeredLayers      = 29
    cudaDevAttrSurfaceAlignment               = 30
    cudaDevAttrConcurrentKernels              = 31
    cudaDevAttrEccEnabled                     = 32
    cudaDevAttrPciBusId                       = 33
    cudaDevAttrPciDeviceId                    = 34
    cudaDevAttrTccDriver                      = 35
    cudaDevAttrMemoryClockRate                = 36
    cudaDevAttrGlobalMemoryBusWidth           = 37
    cudaDevAttrL2CacheSize                    = 38
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39
    cudaDevAttrAsyncEngineCount               = 40
    cudaDevAttrUnifiedAddressing              = 41
    cudaDevAttrMaxTexture1DLayeredWidth       = 42
    cudaDevAttrMaxTexture1DLayeredLayers      = 43
    cudaDevAttrMaxTexture2DGatherWidth        = 45
    cudaDevAttrMaxTexture2DGatherHeight       = 46
    cudaDevAttrMaxTexture3DWidthAlt           = 47
    cudaDevAttrMaxTexture3DHeightAlt          = 48
    cudaDevAttrMaxTexture3DDepthAlt           = 49
    cudaDevAttrPciDomainId                    = 50
    cudaDevAttrTexturePitchAlignment          = 51
    cudaDevAttrMaxTextureCubemapWidth         = 52
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54
    cudaDevAttrMaxSurface1DWidth              = 55
    cudaDevAttrMaxSurface2DWidth              = 56
    cudaDevAttrMaxSurface2DHeight             = 57
    cudaDevAttrMaxSurface3DWidth              = 58
    cudaDevAttrMaxSurface3DHeight             = 59
    cudaDevAttrMaxSurface3DDepth              = 60
    cudaDevAttrMaxSurface1DLayeredWidth       = 61
    cudaDevAttrMaxSurface1DLayeredLayers      = 62
    cudaDevAttrMaxSurface2DLayeredWidth       = 63
    cudaDevAttrMaxSurface2DLayeredHeight      = 64
    cudaDevAttrMaxSurface2DLayeredLayers      = 65
    cudaDevAttrMaxSurfaceCubemapWidth         = 66
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68
    cudaDevAttrMaxTexture1DLinearWidth        = 69
    cudaDevAttrMaxTexture2DLinearWidth        = 70
    cudaDevAttrMaxTexture2DLinearHeight       = 71
    cudaDevAttrMaxTexture2DLinearPitch        = 72
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74
    cudaDevAttrComputeCapabilityMajor         = 75
    cudaDevAttrComputeCapabilityMinor         = 76
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77
    cudaDevAttrStreamPrioritiesSupported      = 78
    cudaDevAttrGlobalL1CacheSupported         = 79
    cudaDevAttrLocalL1CacheSupported          = 80
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82
    cudaDevAttrManagedMemory                  = 83
    cudaDevAttrIsMultiGpuBoard                = 84
    cudaDevAttrMultiGpuBoardGroupID           = 85
c_cudaDeviceAttr = c_int

#cudaFuncCache
class cudaFuncCache(enum.IntEnum):
    cudaFuncCachePreferNone   = 0
    cudaFuncCachePreferShared = 1
    cudaFuncCachePreferL1     = 2
    cudaFuncCachePreferEqual  = 3
c_cudaFuncCache = c_int

#cudaLimit
class cudaLimit(enum.IntEnum):
    cudaLimitStackSize                    = 0x00
    cudaLimitPrintfFifoSize               = 0x01
    cudaLimitMallocHeapSize               = 0x02
    cudaLimitDevRuntimeSyncDepth          = 0x03
    cudaLimitDevRuntimePendingLaunchCount = 0x04
c_cudaLimit = c_int

#cudaSharedMemConfig
class cudaSharedMemConfig(enum.IntEnum):
    cudaSharedMemBankSizeDefault   = 0
    cudaSharedMemBankSizeFourByte  = 1
    cudaSharedMemBankSizeEightByte = 2
c_cudaSharedMemConfig = c_int

##Structures##

CUDA_IPC_HANDLE_SIZE  = 64
    
#struct cudaDeviceProp
#defined in driver_types.h (V7.0) line 1257
class cudaDeviceProp(Structure):
    _fields_ = [('name',                       c_char*256),
                ('totalGlobalMem',             c_size_t),
                ('sharedMemPerBlock',          c_size_t),
                ('regsPerBlock',               c_int),
                ('warpSize',                   c_int),
                ('memPitch',                   c_size_t),
                ('maxThreadsPerBlock',         c_int),
                ('maxThreadsDim',              c_int*3),
                ('maxGridSize',                c_int*3),
                ('clockRate',                  c_int),
                ('totalConstMem',              c_size_t),
                ('major',                      c_int),
                ('minor',                      c_int),
                ('textureAlignment',           c_size_t),
                ('texturePitchAlignment',      c_size_t),
                ('deviceOverlap',              c_int),
                ('multiProcessorCount',        c_int),
                ('kernelExecTimeoutEnabled',   c_int),
                ('integrated',                 c_int),
                ('canMapHostMemory',           c_int),
                ('computeMode',                c_int),
                ('maxTexture1D',               c_int),
                ('maxTexture1DMipmap',         c_int),
                ('maxTexture1DLinear',         c_int),
                ('maxTexture2D',               c_int*2),
                ('maxTexture2DMipmap',         c_int*2),
                ('maxTexture2DLinear',         c_int*3),
                ('maxTexture2DGather',         c_int*2),
                ('maxTexture3D',               c_int*3),
                ('maxTexture3DAlt',            c_int*3),
                ('maxTextureCubemap',          c_int),
                ('maxTexture1DLayered',        c_int*2),
                ('maxTexture2DLayered',        c_int*3),
                ('maxTextureCubemapLayered',   c_int*2),
                ('maxSurface1D',               c_int),
                ('maxSurface2D',               c_int*2),
                ('maxSurface3D',               c_int*3),
                ('maxSurface1DLayered',        c_int*2),
                ('maxSurface2DLayered',        c_int*3),
                ('maxSurfaceCubemap',          c_int),
                ('maxSurfaceCubemapLayered',   c_int*2),
                ('surfaceAlignment',           c_size_t),
                ('concurrentKernels',          c_int),
                ('ECCEnabled',                 c_int),
                ('pciBusID',                   c_int),
                ('pciDeviceID',                c_int),
                ('pciDomainID',                c_int),
                ('tccDriver',                  c_int),
                ('asyncEngineCount',           c_int),
                ('unifiedAddressing',          c_int),
                ('memoryClockRate',            c_int),
                ('memoryBusWidth',             c_int),
                ('l2CacheSize',                c_int),
                ('maxThreadsPerMultiProcessor',c_int),
                ('streamPrioritiesSupported',  c_int),
                ('globalL1CacheSupported',     c_int),
                ('localL1CacheSupported',      c_int),
                ('sharedMemPerMultiprocessor', c_size_t),
                ('regsPerMultiprocessor',      c_int),
                ('managedMemory',              c_int),
                ('isMultiGpuBoard',            c_int),
                ('multiGpuBoardGroupID',       c_int)]

#cudaIpcEventHandle_t
#defined in driver_types.h (V7.0) line 1397
class cudaIpcEventHandle_t(Structure):
    _fields_ = [('reserved',  c_char*CUDA_IPC_HANDLE_SIZE)]

#cudaIpcMemHandle_t
#defined in driver_types.h (V7.0) line 1405
class cudaIpcMemHandle_t(Structure):
    _fields_ = [('reserved',  c_char*CUDA_IPC_HANDLE_SIZE)]

##Opaque types##
class _opaque(ctypes.Structure): pass

cudaEvent_t = POINTER(_opaque)
cudaEvent_t.__name__ = 'cudaEvent_t'

### Modules ###

##1. Device Management##

# cudaError_t cudaChooseDevice ( int* device,
#                                const cudaDeviceProp* prop )
cudaChooseDevice = libcudart.cudaChooseDevice
cudaChooseDevice.restype = cudaError_t
cudaChooseDevice.argtypes = [POINTER(c_int),          # device
                             POINTER(cudaDeviceProp)  # prop
                             ]

# cudaError_t cudaDeviceGetAttribute ( int* value,
#                                      cudaDeviceAttr attr,
#                                      int device )
cudaDeviceGetAttribute = libcudart.cudaDeviceGetAttribute
cudaDeviceGetAttribute.restype = cudaError_t
cudaDeviceGetAttribute.argtypes = [POINTER(c_int),   # value
                                   c_cudaDeviceAttr, # attr
                                   c_int             # device
                                   ]

# cudaError_t cudaDeviceGetByPCIBusId ( int* device,
#                                       const char* pciBusId )
cudaDeviceGetByPCIBusId = libcudart.cudaDeviceGetByPCIBusId
cudaDeviceGetByPCIBusId.restype = cudaError_t
cudaDeviceGetByPCIBusId.argtypes = [POINTER(c_int), # device
                                    c_char_p        # pciBusId
                                    ]

# cudaError_t cudaDeviceGetCacheConfig ( cudaFuncCache** pCacheConfig )
cudaDeviceGetCacheConfig = libcudart.cudaDeviceGetCacheConfig
cudaDeviceGetCacheConfig.restype = cudaError_t
cudaDeviceGetCacheConfig.argtypes = [POINTER(c_cudaFuncCache)  # pCacheConfig
                                     ]

# cudaError_t cudaDeviceGetLimit ( size_t* pValue,
#                                  cudaLimit limit )
cudaDeviceGetLimit = libcudart.cudaDeviceGetLimit
cudaDeviceGetLimit.restype = cudaError_t
cudaDeviceGetLimit.argtypes = [POINTER(c_size_t), # pValue
                               c_cudaLimit        # limit
                               ]

# cudaError_t cudaDeviceGetPCIBusId ( char* pciBusId,
#                                     int len,
#                                     int device )
cudaDeviceGetPCIBusId = libcudart.cudaDeviceGetPCIBusId
cudaDeviceGetPCIBusId.restype = cudaError_t
cudaDeviceGetPCIBusId.argtypes = [c_char_p, # pciBusId
                                  c_int,    # len
                                  c_int     # device
                                  ]

# cudaError_t cudaDeviceGetSharedMemConfig ( cudaSharedMemConfig* pConfig )
cudaDeviceGetSharedMemConfig = libcudart.cudaDeviceGetSharedMemConfig
cudaDeviceGetSharedMemConfig.restype = cudaError_t
cudaDeviceGetSharedMemConfig.argtypes = [POINTER(c_cudaSharedMemConfig) # pConfig
                                         ]

# cudaError_t cudaDeviceGetStreamPriorityRange ( int* leastPriority,
#                                                int* greatestPriority )
cudaDeviceGetStreamPriorityRange = libcudart.cudaDeviceGetStreamPriorityRange
cudaDeviceGetStreamPriorityRange.restype = cudaError_t
cudaDeviceGetStreamPriorityRange.argtypes = [POINTER(c_int), # leastPriority
                                             POINTER(c_int)  # greatestPriority
                                             ]

# cudaError_t cudaDeviceReset ( void )
cudaDeviceReset = libcudart.cudaDeviceReset
cudaDeviceReset.restype = cudaError_t
cudaDeviceReset.argtypes = []

# cudaError_t cudaDeviceSetCacheConfig ( cudaFuncCache cacheConfig )
cudaDeviceSetCacheConfig = libcudart.cudaDeviceSetCacheConfig
cudaDeviceSetCacheConfig.restype = cudaError_t
cudaDeviceSetCacheConfig.argtypes = [c_cudaFuncCache  # cacheConfig
                                     ]

# cudaError_t cudaDeviceSetLimit ( cudaLimit limit,
#                                  size_t value )
cudaDeviceSetLimit = libcudart.cudaDeviceSetLimit
cudaDeviceSetLimit.restype = cudaError_t
cudaDeviceSetLimit.argtypes = [c_cudaLimit, # limit
                               c_size_t    # value
                               ]

# cudaError_t cudaDeviceSetSharedMemConfig ( cudaSharedMemConfig config )
cudaDeviceSetSharedMemConfig = libcudart.cudaDeviceSetSharedMemConfig
cudaDeviceSetSharedMemConfig.restype = cudaError_t
cudaDeviceSetSharedMemConfig.argtypes = [c_cudaSharedMemConfig  # config
                                         ]

# cudaError_t cudaDeviceSynchronize ( void )
cudaDeviceSynchronize = libcudart.cudaDeviceSynchronize
cudaDeviceSynchronize.restype = cudaError_t
cudaDeviceSynchronize.argtypes = []

# cudaError_t cudaGetDevice ( int* device )
cudaGetDevice = libcudart.cudaGetDevice
cudaGetDevice.restype = cudaError_t
cudaGetDevice.argtypes = [POINTER(c_int)  # device
                          ]

# cudaError_t cudaGetDeviceCount ( int* count )
cudaGetDeviceCount = libcudart.cudaGetDeviceCount
cudaGetDeviceCount.restype = cudaError_t
cudaGetDeviceCount.argtypes = [POINTER(c_int)  # count
                               ]

# cudaError_t cudaGetDeviceFlags ( unsigned int* flags )
cudaGetDeviceFlags = libcudart.cudaGetDeviceFlags
cudaGetDeviceFlags.restype = cudaError_t
cudaGetDeviceFlags.argtypes = [POINTER(c_uint)  # flags
                               ]

# cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop,
#                                       int device )
cudaGetDeviceProperties = libcudart.cudaGetDeviceProperties
cudaGetDeviceProperties.restype = cudaError_t
cudaGetDeviceProperties.argtypes = [POINTER(cudaDeviceProp), # prop
                                    c_int                    # device
                                    ]

# cudaError_t cudaIpcCloseMemHandle ( void* devPtr )
cudaIpcCloseMemHandle = libcudart.cudaIpcCloseMemHandle
cudaIpcCloseMemHandle.restype = cudaError_t
cudaIpcCloseMemHandle.argtypes = [c_void_p  # devPtr
                                  ]

# cudaError_t cudaIpcGetEventHandle ( cudaIpcEventHandle_t* handle,
#                                     cudaEvent_t event )
cudaIpcGetEventHandle = libcudart.cudaIpcGetEventHandle
cudaIpcGetEventHandle.restype = cudaError_t
cudaIpcGetEventHandle.argtypes = [POINTER(cudaIpcEventHandle_t), # handle
                                  cudaEvent_t                    # event
                                  ]

# cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle,
#                                   void* devPtr )
cudaIpcGetMemHandle = libcudart.cudaIpcGetMemHandle
cudaIpcGetMemHandle.restype = cudaError_t
cudaIpcGetMemHandle.argtypes = [POINTER(cudaIpcMemHandle_t), # handle
                                c_void_p                     # devPtr
                                ]

# cudaError_t cudaIpcOpenEventHandle ( cudaEvent_t* event,
#                                      cudaIpcEventHandle_t handle )
cudaIpcOpenEventHandle = libcudart.cudaIpcOpenEventHandle
cudaIpcOpenEventHandle.restype = cudaError_t
cudaIpcOpenEventHandle.argtypes = [POINTER(cudaEvent_t), # event
                                   cudaIpcEventHandle_t  # handle
                                   ]

# cudaError_t cudaIpcOpenMemHandle ( void** devPtr,
#                                    cudaIpcMemHandle_t handle,
#                                    unsigned int flags )
cudaIpcOpenMemHandle = libcudart.cudaIpcOpenMemHandle
cudaIpcOpenMemHandle.restype = cudaError_t
cudaIpcOpenMemHandle.argtypes = [POINTER(c_void_p),  # devPtr
                                 cudaIpcMemHandle_t, # handle
                                 c_uint              # flags
                                 ]

# cudaError_t cudaSetDevice ( int device )
cudaSetDevice = libcudart.cudaSetDevice
cudaSetDevice.restype = cudaError_t
cudaSetDevice.argtypes = [c_int  # device
                          ]

# cudaError_t cudaSetDeviceFlags ( unsigned int flags )
cudaSetDeviceFlags = libcudart.cudaSetDeviceFlags
cudaSetDeviceFlags.restype = cudaError_t
cudaSetDeviceFlags.argtypes = [c_uint  # flags
                               ]

# cudaError_t cudaSetValidDevices ( int* device_arr,
#                                   int len )
cudaSetValidDevices = libcudart.cudaSetValidDevices
cudaSetValidDevices.restype = cudaError_t
cudaSetValidDevices.argtypes = [POINTER(c_int), # device_arr
                                c_int           # len
                                ]

##2. Thread Management##
#Not implemented DEPRECATED

##3. Error Handling##

# char* cudaGetErrorName ( cudaError_t error )
cudaGetErrorName = libcudart.cudaGetErrorName
cudaGetErrorName.restype = c_char_p
cudaGetErrorName.argtypes = [c_cudaError_t  # error
                             ]

# char* cudaGetErrorString ( cudaError_t error )
cudaGetErrorString = libcudart.cudaGetErrorString
cudaGetErrorString.restype = c_char_p
cudaGetErrorString.argtypes = [c_cudaError_t  # error
                               ]

# cudaError_t cudaGetLastError ( void )
cudaGetLastError = libcudart.cudaGetLastError
cudaGetLastError.restype = cudaError_t
cudaGetLastError.argtypes = []

# cudaError_t cudaPeekAtLastError ( void )
cudaPeekAtLastError = libcudart.cudaPeekAtLastError
cudaPeekAtLastError.restype = cudaError_t
cudaPeekAtLastError.argtypes = []


##26 Version management##

# ​cudaError_t cudaDriverGetVersion ( int* driverVersion )
cudaDriverGetVersion = libcudart.cudaDriverGetVersion
cudaDriverGetVersion.restype = cudaError_t
cudaDriverGetVersion.argtypes = [POINTER(c_int)]

# cudaError_t cudaRuntimeGetVersion ( int* runtimeVersion ) 
cudaRuntimeGetVersion = libcudart.cudaRuntimeGetVersion
cudaRuntimeGetVersion.restype = cudaError_t
cudaRuntimeGetVersion.argtypes = [POINTER(c_int)]



###Published symbols in libcudart.so.7.0 not implemented yet:

#1

# __cudaInitManagedRuntime
# __cudaInitModule
# __cudaRegisterDeviceFunct
# __cudaRegisterFatBinary
# __cudaRegisterFunction
# __cudaRegisterManagedVar
# __cudaRegisterPrelinkedFa
# __cudaRegisterShared
# __cudaRegisterSharedVar
# __cudaRegisterSurface
# __cudaRegisterTexture
# __cudaRegisterVar
# __cudaUnregisterFatBinary
# cudaArrayGetInfo
# cudaBindSurfaceToArray
# cudaBindTexture
# cudaBindTexture2D
# cudaBindTextureToArray
# cudaBindTextureToMipmappe
# cudaConfigureCall
# cudaCreateChannelDesc
# cudaCreateSurfaceObject
# cudaCreateTextureObject
# cudaDestroySurfaceObject
# cudaDestroyTextureObject
# cudaDeviceCanAccessPeer
# cudaDeviceDisablePeerAcce
# cudaDeviceEnablePeerAcces
# cudaDeviceGetAttribute
# cudaDeviceGetByPCIBusId
# cudaDeviceGetCacheConfig
# cudaDeviceGetLimit
# cudaDeviceGetPCIBusId
# cudaDeviceGetSharedMemCon
# cudaDeviceGetStreamPriori
# cudaDeviceReset
# cudaDeviceSetCacheConfig
# cudaDeviceSetLimit
# cudaDeviceSetSharedMemCon
# cudaDeviceSynchronize


# cudaEventCreate
# cudaEventCreateWithFlags
# cudaEventDestroy
# cudaEventElapsedTime
# cudaEventQuery
# cudaEventRecord
# cudaEventRecord_ptsz
# cudaEventSynchronize
# cudaFree
# cudaFreeArray
# cudaFreeHost
# cudaFreeMipmappedArray
# cudaFuncGetAttributes
# cudaFuncSetCacheConfig
# cudaFuncSetSharedMemConfi
# cudaGLGetDevices
# cudaGLMapBufferObject
# cudaGLMapBufferObjectAsyn
# cudaGLRegisterBufferObjec
# cudaGLSetBufferObjectMapF
# cudaGLSetGLDevice
# cudaGLUnmapBufferObject
# cudaGLUnmapBufferObjectAs
# cudaGLUnregisterBufferObj
# cudaGetChannelDesc
# cudaGetDevice
# cudaGetDeviceCount
# cudaGetDeviceFlags
# cudaGetErrorName
# cudaGetErrorString
# cudaGetExportTable
# cudaGetLastError
# cudaGetMipmappedArrayLeve
# cudaGetSurfaceObjectResou
# cudaGetSurfaceReference
# cudaGetSymbolAddress
# cudaGetSymbolSize
# cudaGetTextureAlignmentOf
# cudaGetTextureObjectResou
# cudaGetTextureObjectResou
# cudaGetTextureObjectTextu
# cudaGetTextureReference
# cudaGraphicsGLRegisterBuf
# cudaGraphicsGLRegisterIma
# cudaGraphicsMapResources
# cudaGraphicsResourceGetMa
# cudaGraphicsResourceGetMa
# cudaGraphicsResourceSetMa
# cudaGraphicsSubResourceGe
# cudaGraphicsUnmapResource
# cudaGraphicsUnregisterRes
# cudaGraphicsVDPAURegister
# cudaGraphicsVDPAURegister
# cudaHostAlloc
# cudaHostGetDevicePointer
# cudaHostGetFlags
# cudaHostRegister
# cudaHostUnregister
# cudaIpcCloseMemHandle
# cudaIpcGetEventHandle
# cudaIpcGetMemHandle
# cudaIpcOpenEventHandle
# cudaIpcOpenMemHandle
# cudaLaunch
# cudaLaunchKernel
# cudaLaunchKernel_ptsz
# cudaLaunch_ptsz
# cudaMalloc
# cudaMalloc3D
# cudaMalloc3DArray
# cudaMallocArray
# cudaMallocHost
# cudaMallocManaged
# cudaMallocMipmappedArray
# cudaMallocPitch
# cudaMemGetInfo
# cudaMemcpy
# cudaMemcpy2D
# cudaMemcpy2DArrayToArray
# cudaMemcpy2DArrayToArray_
# cudaMemcpy2DAsync
# cudaMemcpy2DAsync_ptsz
# cudaMemcpy2DFromArray
# cudaMemcpy2DFromArrayAsyn
# cudaMemcpy2DFromArrayAsyn
# cudaMemcpy2DFromArray_ptd
# cudaMemcpy2DToArray
# cudaMemcpy2DToArrayAsync
# cudaMemcpy2DToArrayAsync_
# cudaMemcpy2DToArray_ptds
# cudaMemcpy2D_ptds
# cudaMemcpy3D
# cudaMemcpy3DAsync
# cudaMemcpy3DAsync_ptsz
# cudaMemcpy3DPeer
# cudaMemcpy3DPeerAsync
# cudaMemcpy3DPeerAsync_pts
# cudaMemcpy3DPeer_ptds
# cudaMemcpy3D_ptds
# cudaMemcpyArrayToArray
# cudaMemcpyArrayToArray_pt
# cudaMemcpyAsync
# cudaMemcpyAsync_ptsz
# cudaMemcpyFromArray
# cudaMemcpyFromArrayAsync
# cudaMemcpyFromArrayAsync_
# cudaMemcpyFromArray_ptds
# cudaMemcpyFromSymbol
# cudaMemcpyFromSymbolAsync
# cudaMemcpyFromSymbolAsync
# cudaMemcpyFromSymbol_ptds
# cudaMemcpyPeer
# cudaMemcpyPeerAsync
# cudaMemcpyToArray
# cudaMemcpyToArrayAsync
# cudaMemcpyToArrayAsync_pt
# cudaMemcpyToArray_ptds
# cudaMemcpyToSymbol
# cudaMemcpyToSymbolAsync
# cudaMemcpyToSymbolAsync_p
# cudaMemcpyToSymbol_ptds
# cudaMemcpy_ptds
# cudaMemset
# cudaMemset2D
# cudaMemset2DAsync
# cudaMemset2DAsync_ptsz
# cudaMemset2D_ptds
# cudaMemset3D
# cudaMemset3DAsync
# cudaMemset3DAsync_ptsz
# cudaMemset3D_ptds
# cudaMemsetAsync
# cudaMemsetAsync_ptsz
# cudaMemset_ptds
# cudaOccupancyMaxActiveBlo
# cudaOccupancyMaxActiveBlo
# cudaPeekAtLastError
# cudaPointerGetAttributes
# cudaProfilerInitialize
# cudaProfilerStart
# cudaProfilerStop

# cudaSetDevice
# cudaSetDeviceFlags
# cudaSetDoubleForDevice
# cudaSetDoubleForHost
# cudaSetValidDevices
# cudaSetupArgument
# cudaStreamAddCallback
# cudaStreamAddCallback_pts
# cudaStreamAttachMemAsync
# cudaStreamAttachMemAsync_
# cudaStreamCreate
# cudaStreamCreateWithFlags
# cudaStreamCreateWithPrior
# cudaStreamDestroy
# cudaStreamGetFlags
# cudaStreamGetFlags_ptsz
# cudaStreamGetPriority
# cudaStreamGetPriority_pts
# cudaStreamQuery
# cudaStreamQuery_ptsz
# cudaStreamSynchronize
# cudaStreamSynchronize_pts
# cudaStreamWaitEvent
# cudaStreamWaitEvent_ptsz

# DEPRECATED Thread Management module
# cudaThreadExit            ** Deprecated, use cudaDeviceReset
# cudaThreadGetCacheConfig  ** Deprecated, use cudaDeviceGetCacheConfig
# cudaThreadGetLimit        ** Deprecated, use cudaDeviceGetLimit
# cudaThreadSetCacheConfig  ** Deprecated, use cudaDeviceSetCacheConfig
# cudaThreadSetLimit        ** Deprecated, use cudaDeviceSetLimit
# cudaThreadSynchronize     ** Deprecated, use cudaDeviceSynchronize

# cudaUnbindTexture
# cudaVDPAUGetDevice
# cudaVDPAUSetVDPAUDevice

