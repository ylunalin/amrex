#
# Generic setup for using PGI
#
CXX = g++
CC  = gcc
FC  = pgfortran
F90 = pgfortran

CXXFLAGS =
CFLAGS   =
FFLAGS   =
F90FLAGS =

########################################################################

pgi_version := $(shell $(CXX) -V 2>&1 | grep 'target')

COMP_VERSION := $(pgi_version)

########################################################################

ifeq ($(DEBUG),TRUE)

  # 2016-12-02: pgi 16.10 doesn't appear to like -traceback together with c++11

  CXXFLAGS += -g -O0 -Mbounds
  CFLAGS   += -g -O0 -Mbounds
  FFLAGS   += -g -O0 -Mbounds -Ktrap=divz,inv -Mchkptr
  F90FLAGS += -g -O0 -Mbounds -Ktrap=divz,inv -Mchkptr

else

  CXXFLAGS += -g -O3
  CFLAGS   += -g -O3
  FFLAGS   += -gopt -fast
  F90FLAGS += -gopt -fast

endif

########################################################################

CXXFLAGS += -std=c++11
CFLAGS   += -std=gnu99

F90FLAGS += -module $(fmoddir) -I$(fmoddir) -Mdclchk -Mnomain -noacc
FFLAGS   += -module $(fmoddir) -I$(fmoddir) -Mextend -Mnomain -noacc

########################################################################

GENERIC_COMP_FLAGS =

ifeq ($(USE_OMP),TRUE)
  CXXFLAGS += -fopenmp
  CFLAGS += -fopenmp
  F90FLAGS += -mp=nonuma -Minfo=mp
  FFLAGS += -mp=nonuma -Minfo=mp
  override XTRALIBS += -lgomp
endif

ifeq ($(USE_ACC),TRUE)
  GENERIC_COMP_FLAGS += -acc -Minfo=acc -ta=nvidia -lcudart -mcmodel=medium
else
  GENERIC_COMP_FLAGS += -noacc
endif


########################################################################

# Because we do not have a Fortran main

ifeq ($(which_computer),$(filter $(which_computer),summit))
override XTRALIBS += -lstdc++ -pgf90libs -L /sw/summitdev/gcc/5.4.0new/lib64/ -latomic
else
override XTRALIBS += -lstdc++ -pgf90libs -latomic
endif

LINK_WITH_FORTRAN_COMPILER ?= $(USE_F_INTERFACES)

