#!/bin/bash

if [ $# == 0 ]
  then
    echo "Usage: ./run_benchmark.sh output_dir"
fi

# EXEC=./main2d.pgi.TPROF.CUDA.ex
EXEC=./main2d.gnu.TPROF.ex
INPUT=inputs_2d
OUTPUT=run.log

mkdir -p $1 && cp {$INPUT,$EXEC,advance_2d.F90} $1 && cd $1

# have changed directory to the benchmark directory 
$EXEC $INPUT &>$OUTPUT


