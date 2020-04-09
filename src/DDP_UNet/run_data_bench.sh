#!/bin/bash

#check number of arguments
if [ "$#" -ne 1 ]; then
    totalranks=1
else
    totalranks=$1
fi

#options
export OMPI_MCA_btl=^openib
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#mpirun
mpirun -np ${totalranks} ${mpioptions} \
       python data_bench_dali_lowmem.py
