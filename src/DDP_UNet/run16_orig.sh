#!/bin/bash -l

#number of ranks
totalranks=16

#hdf file locking 
export HDF5_USE_FILE_LOCKING=FALSE

#mpi stuff
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

# profile
#profilecmd="nsys profile --stats=true --mpi-impl=openmpi --trace=cuda,cublas,nvtx,osrt,mpi -f true -o /data/profiles/orig_pipe/profile_dali_%q{OMPI_COMM_WORLD_RANK}"

#srun
mpirun -np ${totalranks} ${mpioptions} \
     ${profilecmd} $(which python) train_orig.py \
     --yaml_config "config/UNet.yaml" \
     --comm_mode "openmpi-nccl"
