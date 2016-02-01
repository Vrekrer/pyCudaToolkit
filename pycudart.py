def DriverVersion(self):
    version = ctypes.c_int()
    error = cudaDriverGetVersion(version)
    return error, version.value
