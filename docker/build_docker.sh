#!/bin/bash

#some parameters
tag="new_scaler"

cd ..

## Base
#nvidia-docker build -t registry.services.nersc.gov/tkurth/pytorch-cosmo.deps \
#	      --build-arg PYVER=3.7 \
#	      --build-arg PYV=37 \
#	      -f docker/Dockerfile.deps.ubuntu .
#exit

## pytorch
#nvidia-docker build -t registry.services.nersc.gov/tkurth/pytorch-cosmo.pytorch:${tag} \
#	      --build-arg TAG=${tag} \
#	      --build-arg PYVER=3.7 \
#	      --build-arg PYV=37 \
#	      -f docker/Dockerfile.pytorch .
#exit

## dali
#nvidia-docker build -t registry.services.nersc.gov/tkurth/pytorch-cosmo.dali:${tag} \
#	      --build-arg TAG=${tag} \
#	      --build-arg PYVER=3.7 \
#	      --build-arg PYV=37 \
#	      -f docker/Dockerfile.dali .
#exit

# cosmo 3d
for tg in old_scaler new_scaler; do
    nvidia-docker build -t registry.services.nersc.gov/tkurth/pytorch-cosmo:${tg} \
		  --build-arg TAG=${tg} \
		  -f docker/Dockerfile .
done
#docker push registry.services.nersc.gov/tkurth/pytorch-cosmo:latest

#run docker test
#docker run --device=/dev/nvidia-fs0 --workdir "/opt/pytorch/numpy_reader/scripts" -it tkurth/pytorch-numpy_reader:latest ./reader_test.sh
#nvidia-docker run --workdir "/opt/pytorch/numpy_reader/scripts" -it tkurth/pytorch-numpy_reader:latest ./reader_test.sh
