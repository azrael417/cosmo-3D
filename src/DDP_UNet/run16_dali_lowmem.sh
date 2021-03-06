#!/bin/bash -l

#number of ranks
totalranks=16

#hdf file locking 
export HDF5_USE_FILE_LOCKING=FALSE

#mpi stuff
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

# profile
#profilecmd="nsys profile --stats=true --mpi-impl=openmpi --trace=cuda,cublas,nvtx,osrt,mpi -f true -o /data/profiles/dali_pipe/profile_dali_lowmem_amp_threads_nranks${totalranks}_%q{OMPI_COMM_WORLD_RANK}"

#srun
mpirun -np ${totalranks} ${mpioptions} \
     ${profilecmd} $(which python) train_dali_lowmem.py \
     --yaml_config "config/UNet_transpose.yaml" \
     --comm_mode "openmpi-nccl" \
     --no_copy \
     --enable_amp
