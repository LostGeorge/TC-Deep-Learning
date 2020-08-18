import tarfile
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

with tarfile.open('src/HURSAT-goes_v01_1994249N11200_MELE.tar.gz', 'r:gz') as tar:
    f = tar.getmembers()[5]
    nc_file = tar.extractfile(f)
    with netCDF4.Dataset('dummy_fp', mode='r', memory=nc_file.read()) as nc:
        lat = nc.variables['CentLat'][:]
        lon = nc.variables['CentLon'][:]
        wind_speed = nc.variables['WindSpd'][:]
        pressure = nc.variables['CentPrs'][:] # is invalid for this?
        img = nc.variables['irwin'][:].reshape((601, 601))
        print(img[:5, -5:])
        plt.imshow(img)
        plt.show()

