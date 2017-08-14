AMREX_HOME ?= ../../..

DIM = 3

USE_MPI = TRUE

DEBUG = FALSE
DEBUG = TRUE

TINY_PROFILE = FALSE

COMP = gcc

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/LinearSolvers/C_CellMG/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
