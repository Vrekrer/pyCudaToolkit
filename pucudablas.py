import ctypes, platform

#  cuBLAS Library 

if platform.system()=='Microsoft': libcublas = ctypes.windll.LoadLibrary('cublas.dll')
if platform.system()=='Linux':     libcublas = ctypes.cdll.LoadLibrary('libcublas.so')
else:                              libcublas = ctypes.cdll.LoadLibrary('libcublas.so')

