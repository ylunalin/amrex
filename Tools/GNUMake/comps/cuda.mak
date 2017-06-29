#
# Setup for compiling the CUDA version of AMReX with
# CUDA C, not CUDA Fortran
# Assumes you have set USE_CUDA=TRUE, and have
# set the variables PGI_PATH to the root PGI
# directory and CUDA_PATH to the root CUDA directory.
#
CXX = nvcc
CC  = nvcc
FC  = pgfortran
F90 = pgfortran

CXXFLAGS = -Wno-deprecated-gpu-targets -x cu --std=c++11 -ccbin=g++ -O3
CFLAGS   = -Wno-deprecated-gpu-targets -x c -ccbin=gcc -c99 -O3
FFLAGS   =
F90FLAGS =

########################################################################

pgi_version := $(shell $(CXX) -V 2>&1 | grep 'target')

COMP_VERSION := $(pgi_version)

########################################################################

ifeq ($(DEBUG),TRUE)

  # 2016-12-02: pgi 16.10 doesn't appear to like -traceback together with c++11

  CXXFLAGS += -G -Xcompiler='-g -O0 -fno-inline -ggdb -Wall -Wno-sign-compare -ftrapv'
  CFLAGS   += -G -Xcompiler='-g -O0 -fno-inline -ggdb -Wall -Wno-sign-compare -ftrapv'
  FFLAGS   += -g -O0 -Mbounds -Ktrap=divz,inv -Mchkptr
  F90FLAGS += -g -O0 -Mbounds -Ktrap=divz,inv -Mchkptr

else

  CXXFLAGS += -Xcompiler='-g -O3'
  CFLAGS   += -Xcompiler='-g -O3'
  FFLAGS   += -gopt -fast
  F90FLAGS += -gopt -fast

endif

########################################################################


F90FLAGS += -module $(fmoddir) -I$(fmoddir) -Mdclchk
FFLAGS   += -module $(fmoddir) -I$(fmoddir) -Mextend

########################################################################

GENERIC_COMP_FLAGS =

ifeq ($(USE_OMP),TRUE)
  GENERIC_COMP_FLAGS += -mp=nonuma -Minfo=mp
endif

ifeq ($(USE_ACC),TRUE)
  GENERIC_COMP_FLAGS += -acc -Minfo=acc -ta=nvidia -lcudart -mcmodel=medium
else
  GENERIC_COMP_FLAGS += 
endif

ifeq ($(USE_CUDA),TRUE)
  CXXFLAGS += 
  CFLAGS   += 
  FFLAGS   += -Mcuda=cuda8.0 -Mnomain -Mcuda=lineinfo
  F90FLAGS += -Mcuda=cuda8.0 -Mnomain -Mcuda=lineinfo

  override XTRALIBS += -lstdc++
endif

CXXFLAGS += $(GENERIC_COMP_FLAGS)
CFLAGS   += $(GENERIC_COMP_FLAGS)
FFLAGS   += $(GENERIC_COMP_FLAGS)
F90FLAGS += $(GENERIC_COMP_FLAGS)

########################################################################

# Because we do not have a Fortran main

ifeq ($(which_computer),$(filter $(which_computer),summit))
override XTRALIBS += -pgf90libs -L /sw/summitdev/gcc/5.4.0new/lib64/ -latomic
else
override XTRALIBS += -pgf90libs -latomic
endif
