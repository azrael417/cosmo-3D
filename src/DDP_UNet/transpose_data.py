import numpy as np
import h5py as h5

infile='/data1/cosmo_data/DDP_trdat.h5'
outfile='/data1/cosmo_data/DDP_trdat_tra.h5'

print("Loading file {}".format(infile))
with h5.File(infile, 'r') as f:
    Nbody = f["Nbody"][...]
    Hydro = f["Hydro"][...]

Nbody = np.transpose(Nbody, (1,2,3,0))
Hydro = np.transpose(Hydro, (1,2,3,0))

print("Writing transposed data to file {}".format(outfile))
with h5.File(outfile, 'w') as f:
    f.create_dataset("Nbody",  Nbody.shape)
    f.create_dataset("Hydro",  Hydro.shape)
    f["Nbody"][...] = Nbody[...]
    f["Hydro"][...] = Hydro[...]
