#Example of how to use BuildFEAP (with python):

import argparse
from BuildFEAP import SetBuildFEAP
import os
import pdb

parser = argparse.ArgumentParser(description='FEAP Builder')
parser.add_argument('-m', dest='mode',type=str,default='r',nargs='?')
args = parser.parse_args()
pdb.set_trace()
mode = args.mode

# Set FEAPHOME. (can be done outside the script)
os.environ['FEAPHOME8_3']='/Users/wujie/SourceCode/ShearBands'
#/Users/Christina/Desktop/COLIN/Work/feapver83/ver83

# Define FEAP_ARCH
fa_opt='AOpt'
fa_dbg='ADbg'

# Define compile options for FEAP
o_opt='-O3 -ftree-vectorize -Wall'
o_dbg='-g -Wall'
ldopt='-L/usr/X11R6/lib -lX11 -lblas -llapack -lm'# X11 2.7.8
# ldopt='-lblas -llapack -lm'
# ldopt='-l/usr/X11R6/lib -lblas -llapack -lm'
ff='/usr/local/bin/gfortran' #BuildFEAP will set this based on OS but will also give warning
cc='/usr/bin/gcc'      #BuildFEAP will set this based on OS but will also give warning


# Define possible SLEPC_DIR, PETSC_DIR and PETSC_ARCH
sdir=''
pdir='/Users/wujie/SourceCode/petsc-3.5.4'
pfold='parfeap'
pa_opt='gnu-opt'
pa_dbg='gnu-dbg'


# Set build modes
if mode == 'r': # Rebuild
    # Feap
    SetBuildFEAP()
    # ParFEAP
    SetBuildFEAP(petsc_dir=pdir,PFdir=pfold)

elif mode == 'o': # Optimized
    # Feap (optimized)
    SetBuildFEAP(bopt=o_opt,feap_arch=fa_opt,ldopt=ldopt,ff=ff,cc=cc,oflg=False)
    # ParFEAP (optimized)
    SetBuildFEAP(PFdir=pfold,petsc_dir=pdir,petsc_arch=pa_opt,oflg=False)

elif mode == 'g': # FEAP-Debug
    # Feap (debug)
    SetBuildFEAP(bopt=o_dbg,feap_arch=fa_dbg,ldopt=ldopt,ff=ff,cc=cc)
    # ParFEAP (optimized)
    SetBuildFEAP(PFdir=pfold,petsc_dir=pdir,petsc_arch=pa_opt)

elif mode == 'fg': # Full-Debug
    # Feap (debug)
    SetBuildFEAP(bopt=o_dbg,feap_arch=fa_dbg,ldopt=ldopt,ff=ff,cc=cc)
    # ParFEAP (debug)
    SetBuildFEAP(PFdir=pfold,petsc_dir=pdir,petsc_arch=pa_dbg)


elif mode == 'recomp': # Force Recompile
    # Feap
    SetBuildFEAP(Recomp=True,bopt=o_opt,petsc_dir=pdir,feap_arch=fa_opt,ldopt=ldopt,ff=ff,cc=cc,oflg=True)
    # ParFEAP
    SetBuildFEAP(petsc_dir=pdir,PFdir=pfold,Recomp=True,oflg=True)


