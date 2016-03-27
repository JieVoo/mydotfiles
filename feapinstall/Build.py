#Example of how to use BuildFEAP (with python):

import argparse
from BuildFEAP84 import SetBuildFEAP #SetBuildFEAP(PFdir='',petsc_dir='',petsc_arch='',slepc_dir='',feap_arch='',bopt='',ldopt='',cc='',ff='',Recomp=False,x11off='')
import os

parser = argparse.ArgumentParser(description='FEAP Builder')
parser.add_argument('-m', dest='mode',type=str,default='r',nargs='?')
args = parser.parse_args()
mode = args.mode

# Set FEAPHOME. (can be done outside the script)
os.environ['FEAPHOME8_4']='/home/cxc/software/feap84'

# Define FEAP_ARCH 
fa_opt='AOpt'
fa_dbg='ADbg'

# Define compile options for FEAP
o_opt='-O3 -ftree-vectorize -Wall'
o_dbg='-g -Wall'
ldopt='-lX11 -lblas -llapack -lm'
ff='/usr/bin/gfortran' #BuildFEAP will set this based on OS but will also give warning
cc='/usr/bin/gcc'      #BuildFEAP will set this based on OS but will also give warning


# Define possible SLEPC_DIR, PETSC_DIR and PETSC_ARCH
sdir=''
pdir='/home/cxc/software/petsc-3.5.4'
pfold='parfeap'
pa_opt='arch-linux2-c-opt'
pa_dbg='arch-linux2-c-dbg'


# Set build modes
if mode == 'r': # Rebuild
    # Feap
    SetBuildFEAP()
    # ParFEAP
    SetBuildFEAP(PFdir=pfold)

elif mode == 'o': # Optimized
    # Feap (optimized)
    SetBuildFEAP(bopt=o_opt,feap_arch=fa_opt,ldopt=ldopt,ff=ff,cc=cc)
    # ParFEAP (optimized)
    SetBuildFEAP(PFdir=pfold,petsc_dir=pdir,petsc_arch=pa_opt,slepc_dir=sdir)
    
elif mode == 'g': # FEAP-Debug
    # Feap (debug)
    SetBuildFEAP(bopt=o_dbg,feap_arch=fa_dbg,ldopt=ldopt,ff=ff,cc=cc)
    # ParFEAP (optimized)
    SetBuildFEAP(PFdir=pfold,petsc_dir=pdir,petsc_arch=pa_opt,slepc_dir=sdir)
    
elif mode == 'fg': # Full-Debug
    # Feap (debug)
    SetBuildFEAP(bopt=o_dbg,feap_arch=fa_dbg,ldopt=ldopt,ff=ff,cc=cc)
    # ParFEAP (debug)
    SetBuildFEAP(PFdir=pfold,petsc_dir=pdir,petsc_arch=pa_dbg,slepc_dir=sdir)


elif mode == 'recomp': # Force Recompile
    # Feap
    SetBuildFEAP(Recomp=True)
    # ParFEAP
    SetBuildFEAP(PFdir=pfold,Recomp=True)
    

