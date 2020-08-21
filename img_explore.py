import tarfile
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import time

with tarfile.open('src/HURSAT-goes_v01_2015293N16164_TWENTYSIX.tar.gz', 'r:gz') as tar:
    for f in tar:
        nc_file = tar.extractfile(f)
        with netCDF4.Dataset('dummy_fp', mode='r', memory=nc_file.read()) as nc:
            img = nc.variables['irco2'][:][0]
            img = img.filled(fill_value=273)
            print(np.max(img), np.min(img))
            plt.imshow(img)
            plt.show()
            time.sleep(1)


