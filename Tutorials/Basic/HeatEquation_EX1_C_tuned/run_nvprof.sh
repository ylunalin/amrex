#!/bin/bash

if [ $# == 0 ]
  then
    echo "Usage: ./run_nvprof.sh application_name output_dir"
    exit
fi

APPLICATION=$1
OUTPUTDIR=$2
INPUT=inputs_2d
OUTPUT=main.nvprof

mkdir -p $OUTPUTDIR && cp {$INPUT,$APPLICATION,advance_2d.F90} $OUTPUTDIR && cd $OUTPUTDIR
wait

OPTIONS=""
###### OPTIONS for plain text output ######
# OPTIONS="$OPTIONS --kernels "advance_doit_gpu" --metrics flop_count_dp,dram_read_throughput,dram_write_throughput,dram_read_transactions,dram_write_transactions,flop_dp_efficiency,flop_sp_efficiency"
OPTIONS="$OPTIONS --devices 0 --kernels "advance_doit_gpu_shared" --metrics all"
# OPTIONS="$OPTIONS --print-gpu-trace"


###### OPTIONS for post-processing with nvvp ######
# collect the metrics needed by the analysis system for an individual kernel
# OPTIONS="$OPTIONS --devices 0 --kernels "advance_doit_gpu_shared" --analysis-metrics"
# OPTIONS="$OPTIONS --profile-from-start on --export-profile $OUTPUT"

nvprof $OPTIONS $APPLICATION $INPUT &>>run.log
# echo $OPTIONS 
# echo $APPLICATION 
# echo $INPUT


