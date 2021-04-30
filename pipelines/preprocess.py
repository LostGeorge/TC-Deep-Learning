import tensorflow as tf
import numpy as np
import pandas as pd
import glob, os

def get_data(data_dir):
    img_dir = os.path.join(data_dir, 'img')
    image_types = os.listdir(img_dir)
    n_img_types = len(image_types)
    input_dim = 601

    img_dir_globbed = glob.glob(os.path.join(img_dir, 'irwin', '*.npy'))
    dataset = tf.data.Dataset.list_files(img_dir_globbed, shuffle=True)

    wind_df = pd.read_csv(os.path.join(data_dir, 'wind_data_sampled.csv'), index_col=0)
    pressure_df = pd.read_csv(os.path.join(data_dir, 'pressure_data_sampled.csv'), index_col=0)

    def parse_fp(fp):
        img_stack = np.zeros((input_dim, input_dim, n_img_types), dtype=np.int32)
        for j, typ in enumerate(image_types):
            img_stack[:, :, j] = np.load(os.path.join(img_dir, typ, fp))
        
        img_id = int(fp[:6])

        wind_spd = wind_df.loc[img_id, 'wind_speed']
        pressure = pressure_df.loc[img_id, 'atm_pressure']
        
        # latitude, longitude, sensor_num, sat_lat, sat_long
        extraneous_info = wind_df.loc[img_id][1:-1].to_numpy()

        return img_stack, [wind_spd, pressure], extraneous_info

    dataset = dataset.map(parse_fp)
    return dataset

    


