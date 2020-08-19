import tarfile
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

directory = os.listdir('src')

# Counting the number of elements for better space complexity
ct = 0
for fp in directory:
    with tarfile.open('src/' + fp, 'r:gz') as tar:
        ct += len(tar.getnames())

feat_cols = ['latitude', 'longitude', 'sensor_num', 'sat_lat', 'sat_lon']

wind_np = np.zeros((ct, 6))
press_np = np.zeros((ct, 6))
n_valid = 0
valid_channels = ['irwin'] # 'vschn', 'vschn_hires', 'irnir', 'irwvp', 'irspl', 'irco2']
# Note that visible is a bit weird since need to consider darkness
# TODO: create functions for each channel for proper input -> ubyte linear conversions

for fp in directory:
    splts = fp.split('_')
    name = splts[-1][:-7]
    year = splts[-2][:4]
    with tarfile.open('src/' + fp, 'r:gz') as tar:
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
                        img = nc.variables[ch_name][:][0].filled(fill_value=273)
                        img[img < 170] = 170
                        img -= 170
                        small_img = img.astype(np.uint8)
                        path = 'img/' + ch_name + '/' + name + '_' + year + '_' + str(n_valid)
                        np.save(path, small_img)

                    n_valid += 1

wind_np = wind_np[:n_valid]
wind_df = pd.DataFrame(wind_np, columns=feat_cols + ['wind_speed'])
wind_df.to_csv('wind_data.csv')

press_np = press_np[:n_valid]
press_df = pd.DataFrame(press_np, columns=feat_cols + ['atm_pressure'])
press_df.to_csv('pressure_data.csv')
