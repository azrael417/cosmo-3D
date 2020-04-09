#FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.10-py3-devel
FROM nvcr.io/nvidia/pytorch:20.03-py3

#install horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL && \
    HOROVOD_NCCL_INCLUDE=/usr/lib/x86_64-linux-gnu && \
    HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu && \
    pip install --no-cache-dir horovod

#install mpi4py
RUN pip install mpi4py

#install other python stuff necessary
RUN pip install h5py ruamel.yaml

#create /data mount point
RUN mkdir -p /data /tmp/numa0 /tmp/numa1

# copy source code to op
COPY src/DDP_UNet /opt/DDP_UNet
