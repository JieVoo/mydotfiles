
from __future__ import division

import argparse
import shutil, inspect
import os

def gmsh2feap(ifile,ofile):
    '''
    This function converts files from the gmsh format to FEAP's format.
    Arguments:
    
    - *ifile* input file in gmsh format
    - *ofile* output file in FEAP's format
    '''
    #get nummat,num
    #Read from input {{{2
    matid = []
    fi = open(ifile,'r')
    fo = open(ofile,'w')
    with open(ifile,'r') as fi:
        with open(ofile,'w') as fo:
            fo.write('COORdinates\n')
            #advance to start of nodes
            while True:
                if fi.readline().startswith("$Nodes"):
                    fi.readline()
                    break

            #get nodes
            while True:
                line = fi.readline()
                if line.startswith("$EndNodes"):
                    fo.write('           !Blank Termination Record\n')
                    break
                else:
                    s = [float(p) for p in line.split()]
                    s = [s[0],0]+s[1:]
                    oline  = "  %d,%d,%E,%E,%E \n" % tuple(s)
                    fo.write(oline)

            fo.write('ELEMents\n')
            #advance to start of elements
            while True:
                if fi.readline().startswith("$Elements"):
                    fi.readline()
                    break

            #get elements
            while True:
                line = fi.readline()
                if line.startswith("$EndElements"):
                    fo.write('           !Blank Termination Record\n')
                    break
                else:
                    l = line.split()
                    #check for material
                    mm = l[3]
                    if mm not in matid:
                        matid.append(mm)
                    ma = matid.index(mm)+1
                    s = [int(p) for p in l]
                    nrec = s[2]
                    esta = 3+nrec
                    s = [s[0],0,ma]+s[esta:]
                    s = ",".join(str(i) for i in s)	
                    s = '  '+s+'\n'

                    fo.write(s)
                    #2}}}
                    
# Function that returns the folder of the current scriptf
def GetHOME():
    scriptf = os.path.abspath(inspect.stack()[-1][1]) #Current script
    return os.path.dirname(scriptf) #Path of current script

parser = argparse.ArgumentParser(description='MSH2FEAP')
parser.add_argument('-n', dest='name',type=str,default='r',nargs='?')
args = parser.parse_args()
name = args.name

HOME = GetHOME()
if name[-4:]=='.msh':
    rawname=name[:-4]
    Iname=os.path.join(HOME,name)
    Oname=os.path.join(HOME,rawname)
    gmsh2feap(Iname,Oname)
else:
    print 'please insert .msh name with -n ***.msh'
    






