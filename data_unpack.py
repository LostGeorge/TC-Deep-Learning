import tarfile
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

dir_in = sys.argv[1]
dir_out = sys.argv[2]
directory_lst = os.listdir(dir_in)

# Counting the number of elements for better space complexity
ct = 0
for fp in directory_lst:
    with tarfile.open(dir_in + '/' + fp, 'r:gz') as tar:
        ct += len(tar.getnames())

feat_cols = ['latitude', 'longitude', 'sensor_num', 'sat_lat', 'sat_lon']

wind_np = np.zeros((ct, 6))
press_np = np.zeros((ct, 6))
n_valid = 0

# Select what channels you want for images here
valid_channels = ['irwin'] # 'vschn', 'vschn_hires', 'irnir', 'irwvp', 'irspl', 'irco2']
# Note that visible is a bit weird since need to consider darkness

def sat_to_ubyte(img, key):
    if key == 'irwin' or key == 'irnir':
        img = img.filled(fill_value=293) # room temp is best temp
        img[img < 160] = 293 # coldest cloud top record is 163K, so this is safe
        img[img > 313] = 313
        img = (img - 160) * 5 / 3
    elif key == 'vschn' or key == 'vschn_hires':
        img = img.filled(fill_value=0)
        img[img < 0] = 0
        img[img > 0.5] = 0
        img *= 510
    elif key == 'irwvp' or key == 'irco2':
        img = img.filled(fill_value=273)
        img[img < 188] = 273
        img[img > 273] = 273
        img = (img - 188) * 3
    elif key == "irspl": # This seems to be all masked values?
        return np.zeros(img.shape, dtype=np.uint8)

    return img.astype(np.uint8)

for fp in directory_lst:
    splts = fp.split('_')
    name = splts[-1][:-7]
    year = splts[-2][:4]
    with tarfile.open(dir_in + '/' + fp, 'r:gz') as tar:
        for f in tar:
            nc_file = tar.extractfile(f)
            with netCDF4.Dataset('dummy_fp', mode='r', memory=nc_file.read()) as nc:
                pressure_valid = len(nc.variables['CentPrs'][:].compressed()) > 0
                if pressure_valid:
                    lat = nc.variables['CentLat'][:][0]
                    lon = nc.variables['CentLon'][:][0]
                    sens = nc.variables['sss'][:][0]
                    sat_lat = nc.variables['SubSatLat'][:][0]
                    sat_lon = nc.variables['SubSatLon'][:][0]
                    wind = nc.variables['WindSpd'][:][0]
                    pressure = nc.variables['CentPrs'][:][0]
                    
                    wind_np[n_valid] = [lat, lon, sens, sat_lat, sat_lon, wind]
                    press_np[n_valid] = [lat, lon, sens, sat_lat, sat_lon, pressure]
                    
                    for ch_name in valid_channels:
                        img = nc.variables[ch_name][:][0]
                        small_img = sat_to_ubyte(img, key=ch_name)
                        path = '/'.join([dir_out, 'img', ch_name,
                            '_'.join([str(n_valid).zfill(6), name, year])])
                        np.save(path, small_img)

                    n_valid += 1

wind_np = wind_np[:n_valid]
wind_df = pd.DataFrame(wind_np, columns=feat_cols + ['wind_speed'])
wind_df.to_csv(dir_out + '/wind_data.csv')

press_np = press_np[:n_valid]
press_df = pd.DataFrame(press_np, columns=feat_cols + ['atm_pressure'])
press_df.to_csv(dir_out + '/pressure_data.csv')
