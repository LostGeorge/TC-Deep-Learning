import tarfile
import urllib.request
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import time

stream = urllib.request.urlopen('ftp://filsrv.cicsnc.org/kknapp/hursat_goes/HURSAT-goes_v01_2013180N11256_DALILA.tar.gz')

with tarfile.open(fileobj=stream, mode='r:gz') as tar:
    for f in tar:
        nc_file = tar.extractfile(f)
        with netCDF4.Dataset('dummy_fp', mode='r', memory=nc_file.read()) as nc:
            img = nc.variables['irco2'][:][0]
            img = img.filled(fill_value=273)
            print(np.max(img), np.min(img))
            plt.imshow(img)
            plt.show()
            time.sleep(1)


