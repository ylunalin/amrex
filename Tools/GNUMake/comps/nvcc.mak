#
# If we are using CUDA, pull in the gcc compiler first
# and override it as necessary. This is done because the
# nvcc compiler driver does not work optimally with pgc++
# as a host compiler at present.
#

include $(AMREX_HOME)/Tools/GNUMake/comps/gnu.mak

# Force immediate expansion of the GCC defines,
# since after this point GCC will no longer be
# the actual compiler defined in CXX.

DEFINES := $(DEFINES)
MANAGED := --expt-relaxed-constexpr --expt-extended-lambda

ifeq ($(USE_OMP_IO),TRUE)
   OMP_IO_FLAGS :=-fopenmp
else
   OMP_IO_FLAGS :=
endif

CXXFLAGS := $(MANAGED) $(OMP_IO) -Wno-deprecated-gpu-targets -dc -x cu --std=c++11 -ccbin=$(CXX) -Xcompiler='$(CXXFLAGS) $(OMP_IO_FLAGS)'
CFLAGS := $(MANAGED) $(OMP_IO) -Wno-deprecated-gpu-targets -dc -x c -ccbin=$(CC) -Xcompiler='$(CFLAGS) $(OMP_IO_FLAGS)'

HOST_CXX := $(CXX)
HOST_CC := $(CC)

CXX := nvcc
CC := nvcc

override XTRALIBS :=

