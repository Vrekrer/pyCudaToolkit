#Published symbols @7.0 
#readelf -Ds /usr/lib/x86_64-linux-gnu/libcublas.so.7.0

from subprocess import Popen, PIPE
import pyperclip

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

class XX():
    pass

c_types_reps = {'int'           :'c_int',
                'size_t'        :'c_size_t',
                'char'          :'c_char',
                'unsigned int'  :'c_uint',
                'void'          :'',
                'char*'         :'c_char_p',
                'void*'         :'c_void_p'
                }

def pharseFunct(doc):
    FunctData = XX()
    #remove unicode chars
    doc = doc.decode('unicode_escape').encode('ascii', 'ignore')
    #split at "("
    data = doc.rsplit('(')
    #get retType and function Name
    FunctData.retType, FunctData.Name = data[0].strip().split()[-2:]
    #get 
    pars = data[1].rsplit(')')[0].strip().split(',')
    FunctData.pars = [p.rsplit() for p in pars]
    return FunctData
    

def codeFunct(FunctData, libname):
    code = ''
    c_header = '# ' + FunctData.retType + ' ' + FunctData.Name + ' ( '
    lenH = len(c_header) - 1
    for i, p in enumerate(FunctData.pars):
        c_header += ' '.join(p)
        if (i+1) != len(FunctData.pars):
            c_header += ( ',\n#' + lenH*' ' )
        else:
            c_header += ' )'
    code += c_header + '\n'
    code += FunctData.Name + ' = ' + libname + '.' + FunctData.Name + '\n'
    code += FunctData.Name + '.restype = ' + FunctData.retType + '\n'
    args = FunctData.Name + '.argtypes = ['
    lenA = len(args)
    argtypes = []
    argNames = []
    for pars in FunctData.pars:
        if len(pars) == 1:
            argtypes.append(pars[0])
            argNames.append('')
        elif len(pars) == 2:
            argtypes.append(pars[0])
            argNames.append(pars[1])
        elif len(pars) == 3:
            if pars[0] == 'const':
                argtypes.append(pars[1])
            else: 
                argtypes.append(' '.join(pars[:2]))
            argNames.append(pars[2])
    for i, t in enumerate(argtypes):
        if t in c_types_reps.keys():
            argtypes[i] = c_types_reps[t]
        elif (t[:-1] in c_types_reps.keys()) & (t[-1]=='*'):
            argtypes[i] = 'POINTER(' + c_types_reps[t[:-1]] + ')'
        elif t[-1]=='*':
            argtypes[i] = 'POINTER(' + t[:-1] + ')'
        else:
            argtypes[i] = t
    maxArgTypeName = max([len(t) for t in argtypes])+1
    
    for i, argT in enumerate(argtypes):
        args += argT 
        if (i+1) != len(argtypes):
            args += ','
        else:
            args += ' '
        if argNames[i] != '':
            args += ' '*(maxArgTypeName-len(argT))
            args += '# ' + argNames[i]
        args += ( '\n' + lenA*' ' )            
    args += ']\n'
    code += args
    pyperclip.copy(code)

def pharseStructFields(c_code):
    S = XX()
    lines = c_code.splitlines()
    lines = [line.rsplit(';')[0].strip() for line in lines]
    S.datatypes = [l.split()[0] for l in lines]
    S.dataNames = [l.split()[1].rsplit('[')[0] for l in lines]
    S.arraySize = [(l.split()[1]+'[').rsplit('[')[1].rsplit(']')[0] for l in lines]
    S.size = len(S.datatypes)
    S.maxDataNameSize = max([len(a) for a in S.dataNames])
    return S

def codeStruct(sData):
    code = '    _fields_ = ['
    lenH = len(code)
    for i in range(sData.size):
        name_spaces = (sData.maxDataNameSize - len(sData.dataNames[i]))*' '
        code += "('" + sData.dataNames[i] + "'," +name_spaces
        code += c_types_reps[sData.datatypes[i]]
        if sData.arraySize[i] != '':
            code += '*'+sData.arraySize[i]+')'
        else:
            code += ')'
        if (i+1) != sData.size:
            code +=  ',\n' + lenH*' ' 
        else:
            code += ']'
    pyperclip.copy(code)
