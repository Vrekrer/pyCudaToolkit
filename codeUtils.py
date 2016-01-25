#Published symbols @7.0 
#readelf -Ds /usr/lib/x86_64-linux-gnu/libcublas.so.7.0

from subprocess import Popen, PIPE

libDir = '/usr/lib/x86_64-linux-gnu/'

def getSymbolTable(libname):
    (stdout, stderr) = Popen(["readelf", "-Ds",
                              libDir + libname], stdout=PIPE).communicate()
    lines = stdout.splitlines()[3:]
    return [l.split()[8] for l in lines]

def getNotDefined(fileName, base, symbolTable):
    with open(fileName,'r') as pyfile:
        fileText = pyfile.read()
        return [s for s in symbolTable if not(base+'.'+s in fileText)]


# Function to help construct the headers
def header(funct):
    fS = 'cublasS' + funct
    fD = 'cublasD' + funct
    fC = 'cublasC' + funct
    fZ = 'cublasZ' + funct
    for f in [fS, fD, fC, fZ]:
       print '%s = libcublas.%s_v2' % (f, f)
    print 'for funct in [%s, %s, %s, %s]:' % (fS, fD, fC, fZ)
    print '    funct.restype = cublasStatus_t'
    print '    #funct.argtypes = [cublasHandle_t,'
