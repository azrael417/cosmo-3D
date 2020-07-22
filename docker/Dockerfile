ARG TAG

FROM registry.services.nersc.gov/tkurth/pytorch-cosmo.dali:${TAG}

#install cupy
RUN pip install cupy-cuda102

#install mpi4py
RUN pip install mpi4py

#install other python stuff necessary
RUN pip install h5py ruamel.yaml matplotlib zarr

#create /data mount point
RUN mkdir -p /data /tmp/numa0 /tmp/numa1

# copy source code to op
COPY src/DDP_UNet /opt/DDP_UNet
COPY src/DDP_UNet_original /opt/DDP_UNet_original
