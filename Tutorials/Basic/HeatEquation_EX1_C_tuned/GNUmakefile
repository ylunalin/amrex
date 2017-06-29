# AMREX_HOME defines the directory in which we will find all the BoxLib code
# If you set AMREX_HOME as an environment variable, this line will be ignored
AMREX_HOME ?= ../../../

# TINY_PROFILE	= TRUE
TINY_PROFILE	= FALSE
# DEBUG     	= TRUE
DEBUG     	= FALSE


USE_MPI   	= FALSE
USE_OMP   	= FALSE


USE_CUDA_UM	= FALSE
USE_CUDA  	= TRUE
COMP      	= CUDA
# USE_CUDA  	= FALSE
# COMP    	  = gnu


DIM       	= 2

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
